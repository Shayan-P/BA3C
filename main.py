import torch 
import einops as eo

c = 0
g_std_noise = 1e-1

def R(x):
    return (c - x) ** 2

def G(x):
    return 2 * (c - x)

def G_tilde(x):
    return G(x) + g_std_noise * torch.randn_like(x)


def naive_approx(x0, x_ts, g_tilde_ts):
    return g_tilde_ts.mean(dim=0)


def naive_knn_approx(x0, x_ts, g_tilde_ts):
    dist = ((x_ts - x0[None, :]) ** 2).sum(dim=-1)
    best = dist.argmin()
    return g_tilde_ts[best]


def naive_knn_weighted(x0, x_ts, g_tilde_ts):
    dist = ((x_ts - x0[None, :]) ** 2).sum(dim=-1)
    weight = 1/dist
    weight = weight / weight.sum()
    return (weight[:, None] * g_tilde_ts).sum(dim=0)


def our_approx(x0, x_ts, g_tilde_ts):
    """
    shape x0: d
    shape x_ts: n, d
    shape g_tilde_ts: n, d
    """

    x0 = torch.atleast_1d(x0)
    x_ts = torch.atleast_2d(x_ts)

    all_x = torch.cat([x0.unsqueeze(0), x_ts], dim=0)
    n_p1 = all_x.shape[0]

    dist = ((all_x[:, None, :] - all_x[None, :, :]) ** 2).sum(dim=-1)
    alpha = 0.1
    prior_x = torch.exp(-dist * alpha)

    prior_g_arrow_and_tilde = eo.repeat(prior_x, "d1 d2 -> a b d1 d2", a=2, b=2)
    prior_g_arrow_and_tilde[1, 1] += (g_std_noise**2) * torch.eye(n_p1)

    prior_g_arrow_and_tilde = eo.rearrange(prior_g_arrow_and_tilde, "a b d1 d2 -> (a d1) (b d2)")

    sigma_22 = prior_g_arrow_and_tilde[n_p1+1:, n_p1+1:]
    sigma_12 = prior_g_arrow_and_tilde[0, n_p1+1:]

    weight_matrix = sigma_12 @ torch.linalg.inv(sigma_22) 
    print(weight_matrix)
    res = weight_matrix @ g_tilde_ts
    return res


def main():
    x0 = torch.tensor(0.5).unsqueeze(-1)
    x_ts = torch.tensor([-10, -5, 2, 0.53, 10, 11, 12, 13, 14, 15, 16, 1000]).unsqueeze(-1).float()
    g_tilde_ts = G_tilde(x_ts)

    naive_result = naive_approx(x0, x_ts, g_tilde_ts)
    naive_result_knn = naive_knn_approx(x0, x_ts, g_tilde_ts)
    naive_result_knn_weighted = naive_knn_weighted(x0, x_ts, g_tilde_ts)
    our_result = our_approx(x0, x_ts, g_tilde_ts)

    print("naive_result", naive_result)
    print("naive_result_knn", naive_result_knn)
    print("naive_result_knn_weighted", naive_result_knn_weighted)
    print("our_result", our_result)
    print("correct_result", G(x0))


if __name__ == "__main__":
    main()
