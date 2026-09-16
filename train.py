from src.data import ReplayBuffer, make_vec_env
from src.model import RSSM, Decoder, Reward, Continue, Actor, Critic
from src.utils import symlog,symexp,twohot, inv_twohot,make_bins,categorical_kl,categorical_entropy,scale_by_momentum
import yaml
import jax.numpy as jnp
import ninjax as nj
import jax
import optax
import pickle
import time
import os
import numpy as np
import json

with open('dreamerv3.yaml') as f :
    config = yaml.safe_load(f)

B = config['train']['batch_size']     # 16
T = config['train']['batch_length']   # 64
H = config['train']['img_length']     # 15
obs_embedding_dim= 2048 # encoder output, 4*4*128
state_dim = 5120    # [h,z]
lambd_=config['train']['lambd']
sg = lambda x: jax.lax.stop_gradient(x)


twohot_single = jax.vmap(twohot,in_axes=(0, None))
twohot_batched = jax.vmap(twohot_single,in_axes=(0, None))


def swap_first_two(x):
    return jnp.swapaxes(x,0,1)

def lambda_return_step(next_return,transition,lambda_=lambd_):
    reward,value,cont = transition
    returns =reward+cont*((1.-lambda_)*value +lambda_*next_return)
    return returns, returns


def main():
    loss_log = []
    recent_scores = []

    NUM_ENVS = int(config['train'].get('num_envs', 16))
    env = make_vec_env(NUM_ENVS, task=config['env'].get('task'))
    n = int(env.act_space['dim'])

    rssm = RSSM(obs_embed_dim=obs_embedding_dim, n_actions=n, name='rssm')
    act = Actor(dim=n, name='act')
    val = Critic(name='val')
    slow_val = Critic(name='slowval')
    rew = Reward(name='rew')
    dec = Decoder(name='dec')
    con = Continue(name='con')
   #opt
    lr = float(config['opt']['lr'])
    opt = optax.chain(
        optax.adaptive_grad_clip(float(config['opt']['agc'])),
        optax.scale_by_rms(0.999, 1e-20),
        scale_by_momentum(0.9, nesterov=False),
        optax.scale(-lr),
    )
    WARMUP = int(config['opt']['warmup'])
    n_updates = 0

    # bins
    bins =make_bins(v_min=float(config['train']['v_min']),v_max=float(config['train']['v_max']),n_bins=int(config['train']['n_bins']))

    state = {}
    def play(h,z,a_prev_int, x_obs, is_first):
        embedding=rssm.obs_embed(rssm.encoder(x_obs))       # (N, hidden)
        prev_action=jax.nn.one_hot(a_prev_int, n + 1)[..., :-1]
        (h_new, z_new), _ = rssm.observe(embedding, prev_action, (h, z), is_first)
        logits = act(jnp.concatenate([h_new, z_new], axis=-1))
        probs = jax.nn.softmax(logits.astype(jnp.float32), axis=-1)
        probs = probs*0.99 + 0.01 / n
        return (h_new,z_new),jnp.log(probs)

    state = nj.init(play)(state,jnp.zeros((NUM_ENVS, 4096)),jnp.zeros((NUM_ENVS, 1024)),jnp.zeros((NUM_ENVS,),jnp.int32),
                          jnp.zeros((NUM_ENVS, 64, 64, 3)),jnp.ones((NUM_ENVS,)),
                          seed=0)
    play_jit = jax.jit(nj.pure(play))

    state = nj.init(lambda: (
        rssm.prior(jnp.zeros((1,4096))),
        rssm.posterior(jnp.concatenate([jnp.zeros((1,4096)),jnp.zeros((1, 512))],axis=-1)),
        rew(jnp.zeros((1,state_dim))),
        con(jnp.zeros((1,state_dim))),
        dec(jnp.zeros((1,4096)), jnp.zeros((1,1024))),
        val(jnp.zeros((1,state_dim))),
        slow_val(jnp.zeros((1,state_dim))),))(state, seed=2)
    
    new_state = dict(state)
    for k in state.keys():
        if k.startswith('val/'):
            new_state[k.replace('val/','slowval/',1)] = state[k]
    state = new_state

    checkpoint_path ='checkpoints/dreamerv3_latest.pkl'
    checkpoint = None
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path,'rb') as f:
            checkpoint =pickle.load(f)

    replay_dir =config['replay'].get('directory', 'replay')
    buf = ReplayBuffer(size=config['replay']['size'],num_envs=NUM_ENVS,deter=4096,stoch=1024,obs_shape=(64,64,3), 
                       directory=replay_dir,resume=checkpoint is not None)
    
    if checkpoint is not None:
        buf.load_state_dict(checkpoint['replay'])

    ret_norm=jnp.zeros(())
    start_step = 0
    if checkpoint is not None:
        state =checkpoint['state']
        ret_norm =checkpoint['ret_norm']
        start_step = int(checkpoint['step']) + 1
        n_updates = int(checkpoint['n_updates'])
        print(f'Resume from {checkpoint_path} step {start_step}')

    opt_state = opt.init(state) if checkpoint is None else checkpoint['opt_state']


    def forward(x_seq,a_seq,r_seq,d_seq,f_seq,lh0,lz0,ret_norm,n_updates):

        _,((h_seq,z_seq),post_logits_seq) = rssm.observe_seq(x_seq,a_seq,f_seq,(lh0[0],lz0[0]),n)
        prior_flat = rssm.prior(h_seq)
        prior_flat = prior_flat.reshape(*h_seq.shape[:2],32,32)
        prior_probs_raw =jax.nn.softmax(prior_flat.astype(jnp.float32),-1)
        prior_uniform =jnp.ones_like(prior_probs_raw) / 32
        prior_logits =jnp.log(0.01*prior_uniform+0.99*prior_probs_raw)

        post_probs=jax.nn.softmax(post_logits_seq.astype(jnp.float32),-1)
        post_uniform=jnp.ones_like(post_probs)/32
        post_logits_mixed =jnp.log(0.01*post_uniform+0.99*post_probs)

        z_h = jnp.concatenate([h_seq, z_seq],axis=-1)    # (T-1,B,5120)
        reward_pred = rew(z_h)
        cont_pred = con(z_h).squeeze(-1)
        recons_obs = dec(h_seq, z_seq)

        _obs = x_seq[1:].astype(jnp.float32) / 255.0
        recons_loss = optax.l2_loss(recons_obs, _obs).reshape(*r_seq[1:].shape, -1).sum(-1) ## breakthrough....:)
        cont_loss = optax.sigmoid_binary_cross_entropy(
            cont_pred,(1.-d_seq[1:])*config['train']['gamma'])
       
        reward_loss = optax.softmax_cross_entropy(
            reward_pred, twohot_batched(r_seq[1:], bins))
        prediction_loss = recons_loss + cont_loss + reward_loss

        dynamics_loss = jnp.maximum(
            config['train']['free_nats'],
            categorical_kl(sg(post_logits_mixed),prior_logits).sum(-1))
        rep_loss = jnp.maximum(
            config['train']['free_nats'],
            categorical_kl(post_logits_mixed,sg(prior_logits)).sum(-1))

        wm_loss=(config['loss_scales']['rep']*rep_loss.mean()+config['loss_scales']['dyn']*dynamics_loss.mean()+
                 config['loss_scales']['rec']*prediction_loss.mean())

        BT = (T-1)*B
        init_h = h_seq.reshape(BT,4096)
        init_z = z_seq.reshape(BT,1024)

       
        init_logits = act(jnp.concatenate([init_h, init_z],axis=-1))
        init_actions = jax.random.categorical(nj.seed(),init_logits)

        def imagine_policy(full_state):
            logits = act(full_state)
            probs = jax.nn.softmax(logits.astype(jnp.float32),-1)
            probs = probs*0.99+0.01/n
            return jax.nn.one_hot(jax.random.categorical(nj.seed(),jnp.log(probs)),n)

        _,((h_img,z_img),img_actions) = rssm.imagine_seq(imagine_policy,(init_h,init_z),jax.nn.one_hot(init_actions,n),H=H)
        
        img_states =jnp.concatenate([h_img,z_img],axis=-1)   # (H, BT, 5120)

        img_states=sg(img_states)
        img_actions=sg(img_actions)
        init_state=jnp.concatenate([init_h, init_z],axis=-1)   # (BT, 5120)
        full_states=jnp.concatenate([init_state[None], img_states],axis=0)   # (H+1, BT, 5120)
        full_actions=jnp.concatenate([jax.nn.one_hot(init_actions,n)[None],img_actions],axis=0)       # (H+1, BT, n)

        init_rew = r_seq[1:].reshape(BT)
        init_cont = (1.- d_seq[1:]).reshape(BT)

        rewards = inv_twohot(jax.nn.softmax(rew(full_states),-1),bins)   # (H+1, BT)
        cont = jax.nn.sigmoid(con(full_states).squeeze(-1))                # (H+1, BT)
        full_rewards = jnp.concatenate([init_rew[None],rewards[1:]],axis=0)  # (H+1, BT)
        full_cont = jnp.concatenate([init_cont[None],cont[1:]],axis=0)       # (H+1, BT)

        
        full_states = sg(full_states)

        values_logits = val(full_states)                                          # (H+1, BT, 255)
        target_values = inv_twohot(jax.nn.softmax(values_logits,-1),bins)       # (H+1, BT)
        slow_values = inv_twohot(jax.nn.softmax(slow_val(full_states),-1),bins) # (H+1, BT)
       
        weights = sg(jnp.cumprod(full_cont,axis=0))                              # (H+1, BT)

        _, ret = jax.lax.scan(
            lambda_return_step,
            target_values[-1],
            (full_rewards[1:],target_values[1:],full_cont[1:]),
            reverse=True)
        
        img_critic_loss = (optax.softmax_cross_entropy(
            values_logits[:-1], twohot_batched(sg(ret), bins))
            + config['train']['critic_ema_reg'] * optax.softmax_cross_entropy(
                values_logits[:-1], sg(twohot_batched(slow_values[:-1],bins)))
        )*weights[:-1]
        img_critic_loss = img_critic_loss.mean()

        pi_probs = jax.nn.softmax(act(full_states[:-1]),-1)     # (H, BT, n)
        pi_uniform = jnp.ones_like(pi_probs)/n
        pi_probs = 0.01*pi_uniform + 0.99*pi_probs
        pi_ent = -(pi_probs * jnp.log(pi_probs)).sum(-1)          # (H, BT)
        
        adv = sg((ret-target_values[:-1])/jnp.maximum(config['train']['actor_ret_limit'],ret_norm))
       
        logpi = (jnp.log(pi_probs)*full_actions[:-1]).sum(-1)   # (H, BT)
      
        actor_loss = (weights[:-1]*-(adv*logpi+config['train']['actent']*pi_ent)).sum(0).mean()

        replay_value_logits = val(z_h)                                             # (T-1, B, 255)
        replay_slow = inv_twohot(jax.nn.softmax(slow_val(z_h),-1),bins)          # (T-1, B)
        
        replay_target =ret[0].reshape(T-1,B)                                   # (T-1, B)
    
        discount = (1.-d_seq[1:])*config['train']['gamma']                     # (T-1, B)
       
        _, replay_ret = jax.lax.scan(
            lambda_return_step,
            replay_target[-1],
            (r_seq[2:], replay_target[1:],discount[1:]),      
            reverse=True)
        
        replay_critic_loss = ((
            optax.softmax_cross_entropy(
                replay_value_logits[:-1], twohot_batched(sg(replay_ret), bins))
            + config['train']['critic_ema_reg'] * optax.softmax_cross_entropy(
                replay_value_logits[:-1], sg(twohot_batched(replay_slow[:-1], bins)))
        )).mean()

       
        loss = (wm_loss+ config['loss_scales']['value'] *img_critic_loss+ config['loss_scales']['repval'] *replay_critic_loss
                + config['loss_scales']['policy'] *actor_loss)

      
        ret_perc = jnp.percentile(ret.ravel(),jnp.array([5.0,95.0]))
        new_ret_norm = (ret_perc[1] - ret_perc[0]) * config['train']['actor_ret_decay'] \
                       + ret_norm * (1 - config['train']['actor_ret_decay'])

        out_h = jnp.concatenate([lh0[0][None],h_seq],axis=0)    # (T, B, 4096)
        out_z = jnp.concatenate([lz0[0][None],z_seq],axis=0)    # (T, B, 1024)

        all = {'recons': recons_loss.mean(),
               'rew': reward_loss.mean(),
               'con': cont_loss.mean(),
               'dyn': dynamics_loss.mean(),
               'rep': rep_loss.mean(),
               'actor': actor_loss,
               'critic': img_critic_loss,
               'repval': replay_critic_loss,
               'ent': pi_ent.mean(),
               'maxp': pi_probs.max(-1).mean(),
               'scale': new_ret_norm,
               'adv': adv.mean()}
        return loss, (all, (out_h,out_z), new_ret_norm)


    @jax.jit
    def train(state, opt_state, ret_norm, n_updates, key,
              x_seq, a_seq, r_seq, d_seq, f_seq, lh0, lz0):
        def loss_fn(state, ret_norm):
            next_state, (total, (all_, (out_h, out_z), new_rn)) = nj.pure(forward)(
                state,x_seq,a_seq,r_seq,d_seq,f_seq,lh0,lz0,ret_norm,n_updates,seed=key)
            return total, (all_, (out_h, out_z), new_rn)

        (total,(all_, (out_h, out_z),new_rn)), grads =jax.value_and_grad(loss_fn, has_aux=True)(state,ret_norm)
        updates,opt_state = opt.update(grads, opt_state, state)

        scale = jnp.clip(n_updates/WARMUP,0,1)
        state = optax.apply_updates(state, jax.tree.map(lambda x: x*scale,updates))

        new_state = dict(state)
        for k in state.keys():
            if k.startswith('val/'):
                slow_key = k.replace('val/','slowval/',1)
                new_state[slow_key] = (config['train']['critic_ema_tau']*state[k]
                                       +(1-config['train']['critic_ema_tau'])*state[slow_key])
        return new_state, opt_state, total, all_, (out_h, out_z), new_rn

    def save_ckpt(step):
        os.makedirs('checkpoints',exist_ok=True)
        buf.flush()
        ckpt = jax.device_get({'version':3,'state':state,'opt_state':opt_state,
                               'ret_norm': ret_norm, 'n_updates': n_updates,
                               'replay': buf.state_dict(), 'step': step})
        with open(checkpoint_path,'wb') as f:
            pickle.dump(ckpt, f)
        with open('loss_curve.json','w') as f:
            json.dump(loss_log, f)

    obs = env.reset()
    key = jax.random.PRNGKey(0)
    h, z = rssm.init_carry(NUM_ENVS)
    a_prev = np.full(NUM_ENVS,n,dtype=np.int32)
    is_first = np.ones(NUM_ENVS,dtype=np.float32)
    total_scores = np.zeros(NUM_ENVS,dtype=np.float32)
    prev_r = np.zeros(NUM_ENVS,dtype=np.float32)
    prev_done = np.zeros(NUM_ENVS,dtype=np.float32)

    total_steps = int(config['train'].get('steps',300000))
    start_vec = start_step//NUM_ENVS
    Stop = total_steps//NUM_ENVS
    TRAINING_STARTS = B*T

    t_env =t_train=0.0
    for step in range(start_vec,Stop):
        to_env = time.time()
        state, ((h, z), log_probs) = play_jit(state,h,z,a_prev,obs,is_first,seed=step)
        key, subkey = jax.random.split(key)
        a = jax.random.categorical(subkey, log_probs)  # (N,)
        a_np = np.asarray(jax.device_get(a)).reshape(NUM_ENVS)

        next_obs,r,done,terminal_obs = env.step(a_np)

        buf.stage(obs, a_np, prev_r, prev_done, is_first,jax.device_get(h),jax.device_get(z))
        for i in range(NUM_ENVS): # n environments 
            total_scores[i] += float(r[i])
            if done[i]:
                buf.add(i,terminal_obs[i],0,float(r[i]),1.0,0.0,np.zeros(4096, np.float32),np.zeros(1024,np.float32))
                recent_scores.append(float(total_scores[i]))
                avg = sum(recent_scores[-20:])/min(len(recent_scores),20)
                print(f'Step:{step*NUM_ENVS+i} DEAD | Score:{total_scores[i]:.1f} | Avg20:{avg:.1f} |Ep:{len(recent_scores)}')
                total_scores[i] = 0.0

        obs = next_obs
        is_first = done.astype(np.float32)
        a_prev = np.where(done,n,a_np).astype(np.int32)   # reset action = n
        prev_r = np.where(done,0.0,r).astype(np.float32)
        prev_done = np.where(done,0.0,done).astype(np.float32)
        t_env += time.time()-to_env

        if buf.ready(T) and step*NUM_ENVS >TRAINING_STARTS:
            to_train=time.time()
          #update ratio
            target_updates=int(step*NUM_ENVS*config['train']['replay_ratio']/(B*(T-1)))
            while n_updates < target_updates: #
                key, subkey = jax.random.split(key)
                (x_seq,a_seq,r_seq,d_seq,f_seq,lh0,lz0,envs,starts) = buf.sample(B,T)
                ins =[jax.device_put(v) for v in (x_seq,a_seq,r_seq,d_seq,f_seq,lh0,lz0)]
                state, opt_state,loss, all_, (out_h, out_z), ret_norm = train(state,opt_state,ret_norm,n_updates,subkey,*ins)

                buf.write_latents(envs,starts,np.asarray(jax.device_get(out_h)),np.asarray(jax.device_get(out_z)))
                n_updates += 1
            t_train += time.time() -to_train

            if step % max(1,100//NUM_ENVS) == 0:
                env_step = step*NUM_ENVS
                if not np.isfinite(float(loss)):
                    raise RuntimeError(f'non finite loss at {env_step}') #NAN
                print(f"Step:{env_step}/{Stop*NUM_ENVS} | total:{loss:.1f} rec:{float(all_['recons']):.1f} "
                      f"rew:{float(all_['rew']):.2f} dyn:{float(all_['dyn']):.2f} rep:{float(all_['rep']):.2f} "
                      f"act:{float(all_['actor']):.4f} crit:{float(all_['critic']):.4f} rv:{float(all_['repval']):.2f} "
                      f"ent:{float(all_['ent']):.3f} maxp:{float(all_['maxp']):.3f} sc:{float(all_['scale']):.2f} "
                      f"adv:{float(all_['adv']):.4f} | t_env:{t_env:.1f}s t_train:{t_train:.1f}s")
                loss_log.append({'step': env_step, 'loss': float(loss),
                                 **{k: float(v) for k, v in all_.items()}})
                t_env =0.0 ; t_train = 0.0
            if step%max(1,10000//NUM_ENVS) == 0:
                save_ckpt(step*NUM_ENVS)

    save_ckpt(step*NUM_ENVS)


if __name__ == '__main__':
        main()