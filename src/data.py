import os
import numpy as np
import jax
import jax.tree_util
import jax.numpy as jnp

jax.tree_map = jax.tree_util.tree_map
import gymnax


def render_minatar(obs):
    import seaborn as sns
    n_channels = obs.shape[-1]
    cmap = sns.color_palette("cubehelix",n_channels)
    cmap.insert(0,(0,0,0))
    colors = jnp.asarray(cmap, dtype=jnp.float32)        
    numerical_state = jnp.amax(obs*np.reshape(jnp.arange(n_channels)+1,(1,1,-1)),2).astype(jnp.int32)
    return colors[numerical_state]                        


def to_uint8(obs01):
    return (np.asarray(obs01)*255.0).round().clip(0,255).astype(np.uint8)

class GymnaxVec:
    def __init__(self, num_envs=32, seed=0, task='Breakout-MinAtar',
                 render_fn=None, obs_is_pixels=True, size=64):
        self.env,self.params = gymnax.make(task)
        self.num_envs=num_envs
        self.render_fn=render_fn
        self.size=size
        self.key=jax.random.PRNGKey(seed)
        self.states=None
        n = self.env.action_space(self.params).n
        self.act_space = {'dim': n}
        self.obs_is_pixels =obs_is_pixels

        self._reset = jax.jit(jax.vmap(self.env.reset, in_axes=(0,None)))
        self._step = jax.jit(jax.vmap(self.env.step,in_axes=(0,0,0,None)))

        def reset_if_needed(keys,dones,terminal_obs,stepped_states):
            def reset(_):
                obs,states = jax.vmap(self.env.reset, in_axes=(0, None))(keys, self.params)
                return obs, states
            def skip(_):
                return (jnp.zeros_like(terminal_obs),jax.tree_util.tree_map(jnp.zeros_like,stepped_states))
            return jax.lax.cond(jnp.any(dones),reset,skip,operand=None)

        self._reset_if_needed = jax.jit(reset_if_needed)

    def _process_obs(self,obs):
        if self.obs_is_pixels and self.render_fn is not None:
            obs = jax.vmap(self.render_fn)(obs)
        return obs

    def _finish_obs(self,obs):
        if self.obs_is_pixels:
            obs =jax.image.resize(obs.astype(jnp.float32),(self.num_envs,self.size,self.size,3),method='nearest')
        return obs

    def reset(self):
        self.key,k = jax.random.split(self.key)
        keys=jax.random.split(k,self.num_envs)
        obs,self.states=self._reset(keys,self.params)
        obs = self._finish_obs(self._process_obs(obs))
        return to_uint8(jax.device_get(obs))

    def step(self, actions):
        self.key,step_key,reset_key = jax.random.split(self.key,3)
        step_keys =jax.random.split(step_key,self.num_envs)
        reset_keys = jax.random.split(reset_key,self.num_envs)
        acts = jnp.asarray(np.asarray(actions).reshape(self.num_envs),dtype=jnp.int32)

        out =self._step(step_keys,self.states,acts,self.params)
        if len(out)==6:
            terminal_obs,stepped_states,rews,terminated,truncated, _ = out
            dones =jnp.logical_or(terminated,truncated)
        else:
            terminal_obs,stepped_states,rews,dones, _ = out
        reset_obs,reset_states = self._reset_if_needed(reset_keys,dones,terminal_obs,stepped_states)

        def choose_reset(reset_value, stepped_value):
            shape = (self.num_envs,)+(1,)*(stepped_value.ndim-1)
            return jnp.where(dones.reshape(shape),reset_value,stepped_value)

        self.states=jax.tree_util.tree_map(choose_reset,reset_states,stepped_states)
        next_obs = choose_reset(reset_obs,terminal_obs)

        terminal_o = self._finish_obs(self._process_obs(terminal_obs))
        next_o=self._finish_obs(self._process_obs(next_obs))
        next_o,terminal_o, r,d = jax.device_get((next_o,terminal_o,rews,dones))
        return (to_uint8(next_o),np.asarray(r, np.float32),np.asarray(d, bool), to_uint8(terminal_o))

TASKS = {
    # minatar 
    'Breakout-MinAtar':  dict(cls='gymnax', render='minatar',pixels=True),
    'Asterix-MinAtar':   dict(cls='gymnax',render='minatar',pixels=True),
    'Seaquest-MinAtar':  dict(cls='gymnax',   render='minatar',pixels=True),
    'SpaceInvaders-MinAtar': dict(cls='gymnax',render='minatar',pixels=True),
    'Freeway-MinAtar':   dict(cls='gymnax',render='minatar',pixels=True),   
    'CartPole-v1':       dict(cls='gymnax', render=None, pixels=False),
}

def make_vec_env(num_envs=32,seed=0,task='Breakout-MinAtar'):
    spec = TASKS[task]
    render_fn =render_minatar if spec['render'] == 'minatar' else None
    return GymnaxVec(num_envs=num_envs, seed=seed, task=task,render_fn=render_fn, obs_is_pixels=spec['pixels'])


class ReplayBuffer():
    def __init__(self, size=300000, num_envs=1, deter=4096, stoch=1024,obs_shape=(64, 64, 3), obs_dtype=np.uint8, directory=None,
                 resume=False, stage_every=64):
        self.size =int(size)
        self.num_envs =int(num_envs)
        self.capacity =int(np.ceil(self.size/self.num_envs))
        self.deter =int(deter)
        self.stoch =int(stoch)
        self.obs_shape =tuple(obs_shape)
        self.obs_dtype =np.dtype(obs_dtype)
        self.directory =directory
        self.stage_every =int(stage_every)

        if self.directory:
            os.makedirs(self.directory, exist_ok=True)
            mode = 'r+' if resume else 'w+'
            self.x_obs = self._memmap('observations.dat',self.obs_dtype,(self.num_envs,self.capacity,*self.obs_shape),mode)
            self.actions = self._memmap('actions.dat', np.int64,(self.num_envs, self.capacity), mode)
            self.rewards = self._memmap('rewards.dat',np.float32,  (self.num_envs, self.capacity), mode)
            self.dones = self._memmap('dones.dat', np.float32,(self.num_envs,self.capacity), mode)
            self.firsts =   self._memmap('firsts.dat', np.float32, (self.num_envs, self.capacity),mode)
            self.lat_h = self._memmap('latents_h.dat', np.float32, (self.num_envs,self.capacity, self.deter),mode)
            self.lat_z = self._memmap('latents_z.dat', np.float32, (self.num_envs, self.capacity,self.stoch),mode)
        else:
            self.x_obs = np.empty((self.num_envs,self.capacity,*self.obs_shape), self.obs_dtype)
            self.actions = np.empty((self.num_envs, self.capacity), np.int64)
            self.rewards = np.empty((self.num_envs,self.capacity), np.float32)
            self.dones =np.empty((self.num_envs,self.capacity), np.float32)
            self.firsts = np.empty((self.num_envs,self.capacity),np.float32)
            self.lat_h = np.empty((self.num_envs,self.capacity, self.deter), np.float32)
            self.lat_z = np.empty((self.num_envs, self.capacity,self.stoch),np.float32)

        self.idx = np.zeros(self.num_envs, np.int64)
        self.count =np.zeros(self.num_envs,np.int64)
        self._stage = {
            'obs': np.empty((self.num_envs,self.stage_every,*self.obs_shape),self.obs_dtype),
            'act': np.empty((self.num_envs, self.stage_every),np.int64),
            'rew': np.empty((self.num_envs,self.stage_every),np.float32),
            'don':  np.empty((self.num_envs,  self.stage_every),np.float32),
            'fir':np.empty((self.num_envs,self.stage_every), np.float32),
            'lh': np.empty((self.num_envs, self.stage_every, self.deter),np.float32),
            'lz':np.empty((self.num_envs, self.stage_every, self.stoch),np.float32),
        }
        self._stage_n =np.zeros(self.num_envs,np.int64)

    def _memmap(self,name,dtype,shape,mode):
        path = os.path.join(self.directory,name)
        return np.memmap(path, dtype=dtype,mode=mode,shape=shape)

    def _flush_env(self, env_id):
        n = int(self._stage_n[env_id])
        if n == 0:
            return
        slots = (int(self.idx[env_id]) + np.arange(n)) % self.capacity
        self.x_obs[env_id,slots] = self._stage['obs'][env_id, :n]
        self.actions[env_id,slots] = self._stage['act'][env_id, :n]
        self.rewards[env_id,slots] = self._stage['rew'][env_id, :n]
        self.dones[env_id,slots] = self._stage['don'][env_id, :n]
        self.firsts[env_id,slots] = self._stage['fir'][env_id, :n]
        self.lat_h[env_id,slots] = self._stage['lh'][env_id, :n]
        self.lat_z[env_id,slots] = self._stage['lz'][env_id, :n]
        new_idx =(int(self.idx[env_id])+n ) % self.capacity
        self.count[env_id] = min(int(self.count[env_id]) + n,self.capacity)
        self.idx[env_id] = new_idx
        self._stage_n[env_id] = 0

    def stage(self,obs,actions,rewards,dones,firsts,lat_h,lat_z):
        assert len(obs) == self.num_envs
        envs = np.arange(self.num_envs)
        self._stage['obs'][envs,self._stage_n] = np.asarray(obs,self.obs_dtype)
        self._stage['act'][envs,self._stage_n] = np.asarray(actions,np.int64)
        self._stage['rew'][envs,self._stage_n] = np.asarray(rewards,np.float32)
        self._stage['don'][envs,self._stage_n] = np.asarray(dones,np.float32)
        self._stage['fir'][envs,self._stage_n] = np.asarray(firsts,np.float32)
        self._stage['lh'][envs,self._stage_n] = np.asarray(lat_h,np.float32)
        self._stage['lz'][envs,self._stage_n] = np.asarray(lat_z,np.float32)
        self._stage_n += 1
        full = np.flatnonzero(self._stage_n >= self.stage_every)
        for env_id in full:
            self._flush_env(env_id)

    def add(self, env_id, obs, action, reward, done, is_first, lat_h, lat_z):
        self._flush_env(env_id)
        slot = int(self.idx[env_id])
        self.x_obs[env_id, slot] = np.asarray(obs, self.obs_dtype)
        self.actions[env_id, slot] = int(action)
        self.rewards[env_id, slot] = float(reward)
        self.dones[env_id, slot] = float(done)
        self.firsts[env_id, slot] = float(is_first)
        self.lat_h[env_id, slot] = np.asarray(lat_h, np.float32)
        self.lat_z[env_id, slot] = np.asarray(lat_z, np.float32)
        self.idx[env_id] = (slot+1) % self.capacity
        self.count[env_id]=min(int(self.count[env_id]) + 1, self.capacity)

    def flush_all(self):
        for env_id in range(self.num_envs):
            self._flush_env(env_id)

    def ready(self, length=64, minimum=1024):
        enough_steps = (self.count + self._stage_n) >= int(length)
        total = int(self.count.sum() + self._stage_n.sum())
        return bool(total >= max(int(minimum), int(length)) and enough_steps.any())

    def sample(self, B, T):
        self.flush_all()
        B = int(B)
        T = int(T)
        valid_envs = np.flatnonzero(self.count >= T)
        assert len(valid_envs), f'no environment has {T} chronological frames: {self.count.tolist()}'
        envs = np.random.choice(valid_envs, size=B, replace=True)
        gets = np.empty((B, T), np.int64)

        for batch_id, env_id in enumerate(envs):
            count = int(self.count[env_id])
            if count < self.capacity:
                start = np.random.randint(0, count - T + 1)
                gets[batch_id] = start + np.arange(T)
            else:
                start = np.random.randint(0, self.capacity - T + 1)
                gets[batch_id] = (self.idx[env_id] + start + np.arange(T)) % self.capacity

        x = np.empty((T, B, *self.obs_shape), self.obs_dtype)
        a = np.empty((T, B), np.int64)
        r = np.empty((T, B), np.float32)
        d = np.empty((T, B), np.float32)
        f = np.empty((T, B), np.float32)
        lh = np.empty((T, B, self.deter), np.float32)
        lz = np.empty((T, B, self.stoch), np.float32)
        for batch_id, env_id in enumerate(envs):
            g = gets[batch_id]
            x[:, batch_id] = self.x_obs[env_id][g]
            a[:, batch_id] = self.actions[env_id][g]
            r[:, batch_id] = self.rewards[env_id][g]
            d[:, batch_id] = self.dones[env_id][g]
            f[:, batch_id] = self.firsts[env_id][g]
            lh[:, batch_id] = self.lat_h[env_id][g]
            lz[:, batch_id] = self.lat_z[env_id][g]
        return x,a,r,d,f,lh,lz,envs,gets[:, 0]

    def write_latents(self, envs, starts, lat_h, lat_z):
        T = lat_h.shape[0]
        offs = np.arange(T)
        pos = (starts[None,:] + offs[:,None]) % self.capacity  # (T, B)
        self.lat_h[envs[None,:],pos] = lat_h
        self.lat_z[envs[None,:],pos] = lat_z

    # ## persistence
    def state_dict(self):
        self.flush_all()
        return {
            'size':self.size,'num_envs':self.num_envs,
            'capacity':self.capacity,'deter':self.deter,'stoch':self.stoch,
            'idx':np.asarray(self.idx),'count':np.asarray(self.count),
        }

    def load_state_dict(self, state):
        expected = (self.size,self.num_envs,self.capacity,self.deter,self.stoch)
        actual = (int(state['size']),int(state['num_envs']),int(state['capacity']),int(state['deter']),int(state['stoch']))
        assert actual == expected,(actual,expected)
        self.idx[...]=state['idx']
        self.count[...]=state['count']
        self._stage_n[...]=0

    def flush(self):
        self.flush_all()
        for value in (self.x_obs,self.actions,self.rewards,self.dones,self.firsts,self.lat_h,self.lat_z):
            if isinstance(value,np.memmap):
                value.flush()