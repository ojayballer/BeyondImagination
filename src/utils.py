import jax
import jax.numpy as jnp
import optax


def symlog(x,alpha=1.0):  
    return jnp.sign(x)*jnp.log1p(jnp.abs(alpha*x))

def symexp(x,alpha=1.0):
    return jnp.sign(x)*(jnp.expm1(jnp.abs(alpha*x)))


## --- two-hot encoding / decoding ####

def twohot(x,bins):
    x = jnp.clip(x,bins[0],bins[-1])
    k = jnp.sum(bins < x) - 1 # x -->|| k< x < x+1
    def _twohot(k):
        delta = jnp.abs(bins[k+1] - bins[k])
        encoding = jnp.zeros_like(bins, dtype=jnp.float32)
        encoding = encoding.at[k].add((bins[k+1]-x)/delta)
        encoding = encoding.at[k+1].add((x - bins[k])/delta)
        return encoding

    # if x is below the first bin, put all weight on bin 0
    encoding = jax.lax.cond(
        k >= 0,
        _twohot,
        lambda k: jnp.zeros_like(bins).at[0].add(1.),
        k
    )
    return encoding


def inv_twohot(probs,bins,transform=lambda x:x):
    n = probs.shape[-1]
    if n % 2 == 1:
        m = (n - 1)//2
        p1 = probs[...,:m]
        p2 = probs[...,m:m+1]
        p3 = probs[...,m+1:]
        b1 = bins[...,:m]
        b2 = bins[...,m:m+1]
        b3 = bins[...,m+1:]
        wavg = (p2 * b2).sum(-1)+((p1 * b1)[..., ::-1]+(p3 * b3)).sum(-1)
        return transform(wavg)
    else:
        p1 = probs[..., :n // 2]
        p2 = probs[..., n // 2:]
        b1 = bins[..., :n // 2]
        b2 = bins[..., n // 2:]
        wavg = ((p1*b1)[..., ::-1]+(p2 * b2)).sum(-1)
        return transform(wavg)


def make_bins(v_min=-20.0,v_max=20.0,n_bins=255):
    half = jnp.linspace(v_min,0,(n_bins - 1) // 2 + 1,dtype=jnp.float32)
    half = symexp(half)
    bins = jnp.concatenate([half,-half[:-1][::-1]],0)
    return bins


##  lambda returns 
def lambda_return(rewards, dones, values, gamma=0.99, lambd=0.95):
    init_value = jax.lax.cond(
        dones[-1] == 1,
        lambda: 0.,
        lambda: values[-1] )

    def scan_fn(returns, inputs):
        # bellmann backups
        reward,value,cont = inputs
        returns = reward+gamma*cont*((1.-lambd)*value +lambd*returns)
        return returns, returns

    _, returns = jax.lax.scan(
        scan_fn,
        init_value,
        (rewards, values[:-1],(1 - dones)),
        reverse=True)
    return jnp.concatenate([returns,init_value[None]], axis=-1)


#la prop momentum with bias correction 
def scale_by_momentum(beta=0.9,nesterov=False):
    def init_fn(params):
        mu = jax.tree.map(lambda t: jnp.zeros_like(t,jnp.float32),params)
        step = jnp.zeros((),jnp.int32)
        return (step,mu)

    def update_fn(updates,state,params=None):
        step, mu = state
        step = optax.safe_int32_increment(step)
        mu = optax.update_moment(updates,mu,beta,1)
        if nesterov:
            mu_nesterov = optax.update_moment(updates,mu,beta,1)
            mu_hat = optax.bias_correction(mu_nesterov,beta,step)
        else:
            mu_hat = optax.bias_correction(mu,beta,step)
        return mu_hat,(step,mu)

    return optax.GradientTransformation(init_fn,update_fn)


#KL divergence 
def categorical_kl(p_logits, q_logits):
    #KL(p||q)
    p_logits = p_logits.astype(jnp.float32)
    q_logits = q_logits.astype(jnp.float32)
    logp = jax.nn.log_softmax(p_logits,axis=-1)
    logq = jax.nn.log_softmax(q_logits,axis=-1)
    return (jnp.exp(logp)*(logp-logq)).sum(-1)

def categorical_entropy(logits):
    logp = jax.nn.log_softmax(logits, axis=-1)
    return -(jnp.exp(logp)*logp).sum(-1)

#batched twohot 
twohot_single = jax.vmap(twohot, in_axes=(0,None))
twohot_batched = jax.vmap(twohot_single, in_axes=(0,None))