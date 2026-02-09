import torch

from scipy.stats import shapiro
import numpy as np

def sample_gaussian_centered(n=1000, sample_size=100, std_dev=100, shift=0):
    samples = []
    
    while len(samples) < sample_size:
        # Sample from a Gaussian centered at n/2
        sample = int(np.random.normal(loc=n/2+shift, scale=std_dev))
        
        # Check if the sample is in bounds
        if 1 <= sample < n and sample not in samples:
            samples.append(sample)
    
    return samples

def sample_from_quad_center(total_numbers, n_samples, center, pow=1.2):
    while pow > 1:
        # Generate linearly spaced values between 0 and a max value
        x_values = np.linspace((-center)**(1/pow), (total_numbers-center)**(1/pow), n_samples+1)
        #print(x_values)
        #print([x for x in np.unique(np.int32(x_values**pow))[:-1]])
        # Raise these values to the power of 1.5 to get a non-linear distribution
        indices = [0] + [x+center for x in np.unique(np.int32(x_values**pow))[1:-1]]
        if len(indices) == n_samples:
            break
        
        pow -=0.02
    return indices, pow

def sample_from_quad(total_numbers, n_samples, pow=1.2):
    # Generate linearly spaced values between 0 and a max value
    x_values = np.linspace(0, total_numbers**(1/pow), n_samples+1)

    # Raise these values to the power of 1.5 to get a non-linear distribution
    indices = np.unique(np.int32(x_values**pow))[:-1]
    return indices

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x, seq, model, b, timesteps, cache_interval=None, non_uniform=False, pow=None, center=None,  branch=None, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        prv_f = None

        cur_i = 0
        if non_uniform:
            num_slow = timesteps // cache_interval
            if timesteps % cache_interval > 0:
                num_slow += 1
            interval_seq, final_pow = sample_from_quad_center(total_numbers=timesteps, n_samples=num_slow, center=center, pow=pow)
        else:
            interval_seq = list(range(0, timesteps, cache_interval))
            interval = cache_interval
        #print(non_uniform, interval_seq)
        

        slow_path_count = 0
        save_features = []
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            with torch.no_grad():
                if cur_i in interval_seq: #%
                #if cur_i % interval == 0:
                    #print(cur_i, interval_seq)
                    et, cur_f = model(xt, t, prv_f=None,branch=branch)
                    prv_f = cur_f
                    save_features.append(cur_f[0].detach().cpu())
                    slow_path_count+= 1
                else:
                    et, cur_f = model(xt, t, prv_f=prv_f,branch=branch)
                    #quick_path_count+= 1

            #print(i, torch.mean(et) / torch.mean(xt), torch.var(et)/torch.var(xt), torch.mean(et), torch.var(et))

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

            cur_i += 1

    return xs, x0_preds


def compute_spectral_radius(model, xt, t, b, branch=None, power_iter=3,):
    """
    计算当前时间步雅可比矩阵的谱半径 (ρ(J_f))

    参数:
        model: 去噪模型 εθ(x_t, t)
        xt: 当前状态张量 (shape: [B,C,H,W])
        t: 当前时间步张量 (shape: [B])
        branch:
        power_iter: 幂迭代次数

    返回:
        瑞利商 rho_J: 谱半径估计值 (标量)
        对于一般矩阵（包括非埃尔米特矩阵），谱半径𝜌(𝐴)ρ(A) 是矩阵特征值的最大模值。
        瑞利商本身不直接等于谱半径，但通过幂迭代法，瑞利商的绝对值可以逼近谱半径。
    """
    # ====== 数学背景 ======
    # 需要计算 ODE 右项函数 f(x,t) = (x - sqrt(1-α_t)εθ)/sqrt(α_t) - x
    # 雅可比矩阵 J_f = ∂f/∂x = (I/sqrt(α_t) - sqrt(1-α_t)/sqrt(α_t) ∂εθ/∂x) - I
    # 谱半径 ρ(J_f) = max|λ(J_f)|

    B, C, H, W = xt.shape
    xt = xt.detach().requires_grad_(True)  # 启用梯度跟踪

    # ====== 前向计算 f(x,t) ======
    # ====== 前向计算分离计算图 ======
    with torch.no_grad():
        et, _ = model(xt, t, context=None, prv_f=None, branch=branch)
    alpha_t = compute_alpha(b, t.long())

    # ====== 只包装需要微分的操作 ======
    with torch.enable_grad():
        et = et.detach().requires_grad_(True)  # 重新附加梯度
        f = (xt - (1 - alpha_t).sqrt() * et) / alpha_t.sqrt() - xt

    # ====== 幂迭代法 ======
    v = torch.randn_like(xt)  # 初始化随机向量
    v = v / (v.norm() + 1e-6)  # 单位化
    rho_J = 0.0
    for iter in range(power_iter):
        # 最后一次迭代关闭retain_graph
        retain = iter < (power_iter - 1)
        Jv = torch.autograd.grad(
            outputs=f,
            inputs=xt,
            grad_outputs=v,
            retain_graph=retain,  # 关键修改
            create_graph=False,
            only_inputs=True
        )[0]

        # 更新向量
        v = Jv.detach()
        v_norm = v.norm() + 1e-6
        v = v / v_norm

        # 计算瑞利商
        rho_J = (v * Jv).sum().abs().item()

        # 及时释放中间变量
        del Jv
        torch.cuda.empty_cache()

    # ====== 清理计算图 ======
    xt.grad = None
    del f, et
    torch.cuda.empty_cache()

    return rho_J

def find_topk_indices(lst, k):
    """
    找到列表中值最大的 Top-K 个元素的索引
    :param lst: 输入列表
    :param k: 需要找到的最大值的数量
    :return: 最大值对应的索引列表
    """
    import heapq
    # 使用 heapq.nlargest 找到值最大的 K 个元素及其索引
    topk = heapq.nlargest(k, enumerate(lst), key=lambda x: x[1])
    # 返回对应的索引
    return [index for index, value in topk]


def adaptive_generalized_steps_3(x, seq, model, b, timesteps, interval_seq=None, branch=None, quant=False, **kwargs):
    print('runing to function adaptive_generalized_steps_3')

    '''
    cur_i in interval_seq determines to whether the IEC is used
    '''
    model.timesteps = timesteps
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        prv_f = None
        cur_i = 0
        tmp_ets = []
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            # if quant:
            if True:
                time = len(xs) - 1
                if quant:
                    model.set_time(time) # 
                total_steps = len(seq)
                enable_implicit = (cur_i in interval_seq)
                # enable_implicit = False
                if enable_implicit:
                    max_iter = 2  
                else:
                    max_iter = 1  
                tol = 1e-3  # error threshold
                for iter in range(max_iter):  # 
                    # 
                    if iter == 0:
                        if cur_i in interval_seq:  # %
                            et, cur_f = model(xt, t, context=None, prv_f=None, branch=branch)
                            prv_f = cur_f[0]
                        else:
                            et, cur_f = model(xt, t, context=None, prv_f=prv_f, branch=branch)
                        if quant:
                            model.model.time = model.model.time - 1
                        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                        x0_preds.append(x0_t.to('cpu'))
                        c1 = (
                                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                        )
                        c2 = ((1 - at_next) - c1 ** 2).sqrt()
                        # 
                        xt_next_hat = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                        # xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                        # xs.append(xt_next.to('cpu'))
                        # cur_i += 1
                    else:
                        # use xt_next_hat as input
                        if cur_i in interval_seq:  # %
                            et, cur_f = model(xt_next_hat, t, context=None, prv_f=None,
                                              branch=branch)   
                            prv_f = cur_f[0]
                        else:
                            et, cur_f = model(xt_next_hat, t, context=None, prv_f=prv_f,
                                              branch=branch)  
                        if quant:
                            model.model.time = model.model.time - 1

                        # 隐式更新公式（对应理论方程 x_{t-1} = A_t x_t + B_t ε(x_{t-1})）
                        xt_next_new = at_next.sqrt() * ((xt - et * (1 - at).sqrt()) / at.sqrt()) + \
                                      c1 * torch.randn_like(x) + c2 * et

                        # chech if it is converged
                        # if torch.norm(xt_next_new - xt_next_hat) < tol:
                        residual = torch.norm(xt_next_new - xt_next_hat) / (torch.norm(xt_next_hat) + 1e-6)
                        if residual < tol:
                            break

                        
                        gamma = 0.5
                        xt_next_hat = xt_next_hat + (gamma ** iter) * (xt_next_new - xt_next_hat)
                if quant:
                    model.model.time = model.model.time + 1
                # take output
                xs.append(xt_next_hat.to('cpu'))
                cur_i += 1
            else:
                pass
                if cur_i in interval_seq:  # %
                    et, cur_f = model(xt, t, context=None, prv_f=None, branch=branch)
                    prv_f = cur_f[0]
                else:
                    et, cur_f = model(xt, t, context=None, prv_f=prv_f, branch=branch)
                # tmp_ets.append(et.detach().cpu().numpy())
                x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                x0_preds.append(x0_t.to('cpu'))
                c1 = (
                        kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
                )
                c2 = ((1 - at_next) - c1 ** 2).sqrt()
                # print(c1, c2)
                xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
                xs.append(xt_next.to('cpu'))
                cur_i += 1

    return xs, x0_preds



def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
