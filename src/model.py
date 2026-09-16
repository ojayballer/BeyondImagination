import jax.numpy as jnp 
import ninjax as nj
import flax.linen as nn 
import jax 
import functools
from src.utils import symlog,symexp

ACTIVATIONS = {
    'none' : lambda x: x,
    'relu' : jax.nn.relu,
    'silu' : jax.nn.silu,
    'elu' : jax.nn.elu,
    'tanh' : jnp.tanh,
    'sigmoid' : jax.nn.sigmoid,
    'softmax' : jax.nn.softmax,
    'symlog' : symlog,
    'symexp' : symexp,
}
class BlockLinear(nj.Module):
    units : int
    groups : int = 8
    norm : bool = False
    act : str = 'none'
    def __init__(self):
        assert self.units % self.groups == 0
        if self.norm:
            Norm = nj.FromFlax(functools.partial(nn.RMSNorm,dtype=jnp.bfloat16,epsilon=1e-4))
            self.norm_mod = Norm(name='block_norm')

    def __call__(self,inputs):
        in_per_group = inputs.shape[-1]
        shape = (self.groups,in_per_group,self.units//self.groups)

        def init_kernel(shape):
            initializer = nn.initializers.variance_scaling(1.0, 'fan_in','truncated_normal')
            return initializer(nj.seed(), shape, jnp.float32)

        kernel = self.value('kernel',init_kernel,shape)
        bias = self.value('bias',lambda shape: jnp.zeros(shape,jnp.float32),(self.units,))
        kernel = kernel.astype(inputs.dtype)
        bias = bias.astype(inputs.dtype)
        output = jnp.einsum('...gi,gio->...go', inputs, kernel)
        output = output.reshape(*output.shape[:-2], self.units) + bias # merge groups
        if self.norm:
            output = self.norm_mod(output)
        return ACTIVATIONS[self.act](output)


class BlockGRUCell(nj.Module):
    input_dim : int
    hidden_dim : int
    groups : int = 8
    def __init__(self):
        self.linear_in = BlockLinear(units=self.hidden_dim,groups=self.groups,norm=True, act='silu', name='linear_in')
        self.linear_gru = BlockLinear(units=3*self.hidden_dim, groups=self.groups, norm=False, act='none', name='linear_gru')

    def __call__(self,x,h):
        batch_dims = x.shape[:-1]
        h_groups = jnp.reshape(h, (*batch_dims, self.groups, self.hidden_dim // self.groups))
        x_groups = jnp.repeat(x[..., None, :], self.groups, axis=-2)
        gate_in = jnp.concatenate([h_groups, x_groups], axis=-1) 
        hidden = self.linear_in(gate_in) # (..., hidden_dim) merged, normed, silu
        hidden = jnp.reshape(hidden, (*batch_dims, self.groups, -1))
        gates = self.linear_gru(hidden)
        gates = jnp.reshape(gates,(*batch_dims, self.groups, -1))
        r, z, n = [jnp.reshape(g,(*batch_dims, -1)) for g in jnp.split(gates,3,axis=-1)]
        r = jax.nn.sigmoid(r)
        z = jax.nn.sigmoid(z)
        n = jnp.tanh(r * n)
        h = (1.0-z)*n+z*h
        return h
    
class MLP(nj.Module):
    input_dim : int
    output_dim : int
    hidden_dims : tuple = ()
    activation : str = 'silu'
    normalize : bool = True
    outact : str = 'none'
    zero_out_init : bool = False
    bias_init : float = 0.0
    def __init__(self):
        Dense = nj.FromFlax(functools.partial(nn.Dense,dtype=jnp.bfloat16))
        Dense32 = nj.FromFlax(nn.Dense)
        Norm = nj.FromFlax(functools.partial(nn.RMSNorm,dtype=jnp.bfloat16,epsilon=1e-4))
        layers = []
        for h in self.hidden_dims:
            layers.append(Dense(h,name=f'linear_{len(layers)}'))
            if self.normalize:
                layers.append(Norm(name=f'norm_{len(layers)}'))
            layers.append(ACTIVATIONS[self.activation])
        if self.zero_out_init:
            layers.append(Dense32(self.output_dim,kernel_init=nn.initializers.zeros,name='out'))
        elif self.bias_init > 0:
            layers.append(Dense32(self.output_dim,bias_init=nn.initializers.constant(self.bias_init),name='out'))
        else:
            layers.append(Dense32(self.output_dim,name='out'))
        layers.append(ACTIVATIONS[self.outact])
        self.layers = layers

    def __call__(self,x):
        for l in self.layers:
            x = l(x)
        return x

class Sequence(nj.Module):
    hidden : int = 512
    deter : int = 4096
    n_actions : int = 6
    stoch : int = 1024
    def __init__(self):
        
        self.embed_action = MLP(input_dim=self.n_actions,output_dim=self.hidden,hidden_dims=(self.hidden,), normalize=True,name='embed_action')
        self.embed_latent = MLP(input_dim=self.stoch,output_dim=self.hidden,hidden_dims=(self.hidden,), normalize=True,name='embed_latent')
        self.embed_memory = MLP(input_dim=self.deter, output_dim=self.hidden,
                                hidden_dims=(self.hidden,), normalize=True,
                                name='embed_memory')
        self.cell = BlockGRUCell(input_dim=3*self.hidden,hidden_dim=self.deter,groups=8,name='blockgru')

    def __call__(self,prev_action,prev_stoch,prev_deter):
        a = self.embed_action(prev_action)
        z = self.embed_latent(prev_stoch)
        h_mix = self.embed_memory(prev_deter)
        embed = jnp.concatenate([h_mix,z,a],axis=-1)  
        curr_deter = self.cell(embed,prev_deter)
        curr_deter = curr_deter.astype(jnp.float32)
        return curr_deter
    

class Encoder(nj.Module) :  
    hidden :int = 1024 
    def __init__(self) :
        Conv = nj.FromFlax(functools.partial(nn.Conv, dtype=jnp.bfloat16))
        Norm = nj.FromFlax(functools.partial(nn.RMSNorm, dtype=jnp.bfloat16, epsilon=1e-4))
        self.conv1=Conv(features=64,kernel_size=(5,5),name='conv1_enc')
        self.conv2 =Conv(features=96,kernel_size=(5,5),name='conv2_enc')
        self.conv3=Conv(features=128,kernel_size=(5,5),name='conv3_enc')
        self.conv4=Conv(features=128,kernel_size=(5,5),name='conv4_enc')
        self.norm1=Norm(name='norm1_enc')
        self.norm2=Norm(name='norm2_enc')
        self.norm3=Norm(name='norm3_enc')
        self.norm4 =Norm(name ='norm4_enc')
    
    def __call__(self,x_obs) : # input image shape = 64*64
         ##  e_t, 64->32->16->8->4 using maxpool
         x_obs=x_obs.astype(jnp.float32)/255.0-0.5
         output = self.conv1(x_obs)
         B,H,W,C = output.shape
         output = jnp.reshape(output,(B,H//2,2,W//2,2,C)).max(axis=(2,4))
         output=self.norm1(output) # norm1  
         output= jax.nn.silu(output)
         output = self.conv2(output)
         B,H,W,C = output.shape
         output = jnp.reshape(output,(B,H//2,2,W//2,2,C)).max(axis=(2,4))
         output =self.norm2(output)
         output= jax.nn.silu(output)
         output = self.conv3(output)
         B,H,W,C = output.shape
         output = jnp.reshape(output,(B,H//2,2,W//2,2,C)).max(axis=(2,4))
         output =self.norm3(output)
         output= jax.nn.silu(output)
         output=self.conv4(output)
         B,H,W,C = output.shape
         output = jnp.reshape(output,(B,H//2,2,W//2,2,C)).max(axis=(2,4))
         output= self.norm4(output)
         output = jax.nn.silu(output)
         flatten = jnp.reshape(output,(output.shape[0],-1))
         output = flatten
         return flatten  # e_t         


class Decoder(nj.Module):
    def __init__(self): 
        Conv=nj.FromFlax(functools.partial(nn.Conv, dtype=jnp.bfloat16))
        Norm=nj.FromFlax(functools.partial(nn.RMSNorm, dtype=jnp.bfloat16, epsilon=1e-4))
        Dense=nj.FromFlax(functools.partial(nn.Dense, dtype=jnp.bfloat16))
        Convout=nj.FromFlax(nn.Conv)
        self.conv1=Conv(features=128,kernel_size=(5,5),padding='SAME',name='conv_1')
        self.conv2=Conv(features=96,kernel_size=(5,5),padding='SAME',name='conv_2')
        self.conv3=Conv(features=64,kernel_size=(5,5),padding='SAME',name='conv_3')
        self.conv4=Convout(features=3,kernel_size=(5,5),padding='SAME',name='conv_out') # ->>to original channel
       
        self.norm1=Norm(name='norm_1_dec')
        self.norm2=Norm(name='norm_2_dec')
        self.norm3=Norm(name='norm_3_dec')
        self.proj_deter = BlockLinear(units=2048,groups=8,name='proj_deter')
        self.proj_stoch1 = Dense(1024, name='proj_stoch1')
        self.norm_stoch = Norm(name='norm_stoch')
        self.proj_stoch2 = Dense(2048, name='proj_stoch2')
        self.norm_joint =Norm(name="norm_joint") 
        
    def __call__(self,deter,stoch): 
        lead = deter.shape[:-1]
        deter = jnp.reshape(deter,(-1,deter.shape[-1])).astype(jnp.bfloat16)
        stoch = jnp.reshape(stoch,(-1,stoch.shape[-1])).astype(jnp.bfloat16)

        output_deter =self.proj_deter(jnp.reshape(deter,(*deter.shape[:-1],8,-1)))
        output_deter=jnp.reshape(output_deter,(output_deter.shape[0],8,4,4,16))
        output_deter=jnp.transpose(output_deter,(0,2,3,1,4))
        output_deter=jnp.reshape(output_deter,(output_deter.shape[0],4,4,128))
        output_stoch= self.proj_stoch1(stoch)
        output_stoch=jax.nn.silu(self.norm_stoch(output_stoch))
        output_stoch = self.proj_stoch2(output_stoch)
        output_stoch=jnp.reshape(output_stoch,(output_stoch.shape[0],4,4,128))

        output= output_stoch+output_deter
        output =self.norm_joint(output)
        output=jax.nn.silu(output)
        output = jnp.repeat(jnp.repeat(output, 2, axis=-2),2,axis=-3)
        output = jax.nn.silu(self.norm1(self.conv1(output)))
        output= jnp.repeat(jnp.repeat(output, 2, axis=-2), 2,axis=-3)
        output = jax.nn.silu(self.norm2(self.conv2(output)))
        output = jnp.repeat(jnp.repeat(output, 2, axis=-2), 2,axis=-3)
        output = jax.nn.silu(self.norm3(self.conv3(output)))
        output= jnp.repeat(jnp.repeat(output, 2, axis=-2), 2,axis=-3)
        output =self.conv4(output)
        output =jax.nn.sigmoid(output)
        return jnp.reshape(output,(*lead,64,64,3))


# Recurrent State Space Model 
class RSSM(nj.Module):
        hidden :int =512
        stoch : int =32
        classes : int =32
        deter : int =4096
        
        def __init__(self,obs_embed_dim,n_actions):
           self.obs_embed_dim =obs_embed_dim
           self.n_actions=n_actions
           self.sequence=Sequence(hidden=self.hidden,deter=self.deter, n_actions=self.n_actions,
                                  stoch=self.stoch*self.classes,
                                  name="SequenceModel")
           self.prior=MLP(input_dim=self.deter,output_dim=self.stoch*self.classes, hidden_dims=(self.hidden,self.hidden),
                          normalize=True,
                          name="DynamicModel")
           self.posterior=MLP(input_dim=self.deter+self.hidden, 
                              output_dim=self.stoch*self.classes,
                              hidden_dims=(self.hidden,),
                              normalize=True,
                              name="RepresentationModel")
           self.obs_embed=MLP(input_dim=self.obs_embed_dim,
                              output_dim=self.hidden,
                              hidden_dims=(self.hidden,self.hidden),
                              normalize=True,
                              name="ObsEmbed")
           self.encoder=Encoder(name="Encoder")

        def init_carry(self,batch_size=1):
            return (jnp.zeros((batch_size,self.deter)),
                    jnp.zeros((batch_size,self.stoch*self.classes)))

        def observe(self,embedding,prev_action,prev_latent,is_first):
            h_prev,z_prev = prev_latent

            prev_action = jax.lax.select(
                jnp.broadcast_to(is_first[...,None],shape=prev_action.shape).astype(jnp.bool),
                jnp.zeros_like(prev_action),
                prev_action
            )
            h = jnp.where(is_first[...,None]>0.5, jnp.zeros_like(h_prev), h_prev)
            z = jnp.where(is_first[...,None]>0.5, jnp.zeros_like(z_prev), z_prev)

           
            h_t = self.sequence(prev_action,z,h)
            posterior_logits = self.posterior(jnp.concatenate([h_t,embedding],axis=-1))
            posterior_logits = jnp.reshape(posterior_logits,(posterior_logits.shape[0],self.stoch,self.classes))

            raw_probs = jax.nn.softmax(posterior_logits.astype(jnp.float32),axis=-1)
            uniform = jnp.ones_like(raw_probs)*(1.0/self.classes)
            mixed = raw_probs*0.99 + uniform*0.01
            key = nj.seed()
            idx = jax.random.categorical(key,jnp.log(mixed),axis=-1)
            one_hot = jax.nn.one_hot(idx,self.classes)
            z_t = jax.lax.stop_gradient(one_hot)+raw_probs-jax.lax.stop_gradient(raw_probs)
            z_t = jnp.reshape(z_t,(z_t.shape[0],-1))
            return (h_t,z_t) ,posterior_logits

        def observe_seq(self,x_seq,actions_seq,is_first_seq,prev_latent,n_actions):
            T,B,H,W,C = x_seq.shape
            flattened_images=jnp.reshape(x_seq,(T*B,H,W,C))
            flattened_embeddings = self.encoder(flattened_images)
            flat_embed = self.obs_embed(flattened_embeddings)
            dim=flat_embed.shape[-1]
            embeds_seq = jnp.reshape(flat_embed,(T,B,dim))

            def scan_fn(carry, inputs):
                prev_latent, prev_action_int = carry
                embed_obs, action_int, is_first = inputs
                a_prev_onehot = jax.nn.one_hot(prev_action_int,n_actions+1)[...,:-1]
                full_state ,posterior_logits = self.observe(embed_obs,a_prev_onehot,prev_latent,is_first)
                return (full_state ,action_int), (full_state ,posterior_logits)
            return nj.scan(scan_fn,(prev_latent ,actions_seq[0]),(embeds_seq[1:],actions_seq[1:],is_first_seq[1:])) 
          
        def imagine(self,prev_action,prev_latent):
            h_prev,z_prev = prev_latent
            h_t = self.sequence(prev_action,z_prev,h_prev)
            prior_logits = self.prior(h_t)
            prior_logits = jnp.reshape(prior_logits,(prior_logits.shape[0],self.stoch,self.classes))

            raw_probs = jax.nn.softmax(prior_logits.astype(jnp.float32),axis=-1)
            uniform = jnp.ones_like(raw_probs)*(1.0/self.classes)
            mixed = raw_probs*0.99 + uniform*0.01
            key = nj.seed()
            idx = jax.random.categorical(key,jnp.log(mixed),axis=-1)
            one_hot = jax.nn.one_hot(idx,self.classes)
            z_t = jax.lax.stop_gradient(one_hot)+raw_probs-jax.lax.stop_gradient(raw_probs)
            z_t = jnp.reshape(z_t,(z_t.shape[0],-1))
            return (h_t,z_t) ,prior_logits

        def imagine_seq(self,policy,prev_latent,prev_action_onehot,H=15):
            def scan_fn(carry,_):
                latent, action_onehot = carry
                full_state ,prior_logits = self.imagine(action_onehot,latent)
                new_action = policy(jnp.concatenate([full_state[0],full_state[1]],axis=-1))
                return (full_state ,new_action), (full_state ,new_action)
            return nj.scan(scan_fn,(prev_latent,prev_action_onehot),(),length=H)

class Reward(nj.Module):
         hidden : int = 512
         bins : int = 255
         state_dim : int = 5120
         def __init__(self):
             self.mlp = MLP(input_dim=self.state_dim,output_dim=self.bins,hidden_dims=(self.hidden,),normalize=True,zero_out_init=True,name='reward_mlp')

         def __call__(self, full_state):
             return self.mlp(full_state)

class Continue(nj.Module):
        hidden : int = 512
        state_dim : int = 5120
        def __init__(self):
             self.mlp = MLP(input_dim=self.state_dim,output_dim=1,hidden_dims=(self.hidden,),normalize=True,name='cont_mlp')

        def __call__(self, full_state):
             return self.mlp(full_state)


class Actor(nj.Module):
        hidden : int = 512
        state_dim : int = 5120
        dim : int = 6
        def __init__(self):
             self.mlp = MLP(input_dim=self.state_dim,output_dim=self.dim,hidden_dims=(self.hidden,self.hidden,self.hidden),normalize=True,name='actor_mlp')

        def __call__(self, full_state):
             return self.mlp(full_state)


class Critic(nj.Module):
        hidden : int = 512
        bins : int = 255
        state_dim : int = 5120
        def __init__(self):
             self.mlp = MLP(input_dim=self.state_dim,output_dim=self.bins,hidden_dims=(self.hidden,self.hidden,self.hidden),
                            normalize=True,zero_out_init=True,name='critic_mlp')

        def __call__(self, full_state):
             return self.mlp(full_state)