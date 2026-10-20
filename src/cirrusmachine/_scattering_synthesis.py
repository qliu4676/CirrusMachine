"""Image-increment synthesis adapter for Cheng's estimators.

The upstream filters and statistics are used unchanged. These two entry points
retain our seeded perturbation around the initial image and upstream API fixes.
Upstream source: https://github.com/SihaoCheng/scattering_transform
"""

import time

import numpy as np
import torch
from ._vendor.scattering import (
    AlphaScattering2d_cov,
    Bispectrum_Calculator,
    Scattering2d,
    func_hist,
    func_hist_j,
    func_phi4,
    func_phi4_j,
    get_power_spectrum,
    get_random_data,
)


def synthesis(
    estimator_name,
    target,
    image_init=None,
    image_ref=None,
    image_b=None,
    J=None,
    L=4,
    M=None,
    N=None,
    l_oversampling=1,
    frequency_factor=1,
    mode="image",
    optim_algorithm="LBFGS",
    steps=300,
    learning_rate=0.2,
    device="gpu",
    wavelets="morlet",
    seed=None,
    if_large_batch=False,
    C11_criteria=None,
    normalization="P00",
    precision="single",
    print_each_step=False,
    s_cov_func=None,
    s_cov_func_params=[],
    Fourier=False,
    target_full=None,
    ps=False,
    ps_bins=None,
    ps_bin_type="log",
    bi=False,
    bispectrum_bins=None,
    bispectrum_bin_type="log",
    phi4=False,
    phi4_j=False,
    hist=False,
    hist_factor=1,
    hist_j=False,
    hist_j_factor=1,
    ensemble=False,
    N_ensemble=1,
    N_synthesis=1,
    alpha_init=0.01,
    reference_P00=None,
    pseudo_coef=1,
    remove_edge=False,
):
    """
    the estimator_name can be 's_mean', 's_mean_iso', 's_cov', 's_cov_iso', 'alpha_cov',
    the C11_criteria is the condition on j1 and j2 to compute coefficients, in addition to the condition that j2 >= j1.
    Use * or + to connect more than one condition.
    """
    if not torch.cuda.is_available():
        device = "cpu"
    np.random.seed(seed)
    if C11_criteria is None:
        C11_criteria = "j2>=j1"
    if mode == "image":
        _, M, N = target.shape
        print("input_size: ", target.shape)

    # set initial point of synthesis
    if image_init is None:
        if mode == "image":
            if not ensemble:
                image_init = np.random.normal(
                    target.mean((-2, -1))[:, None, None],
                    target.std((-2, -1))[:, None, None],
                    (target.shape[0], M, N),
                )
            else:
                image_init = np.random.normal(
                    target.mean(), target.std(), (N_ensemble, M, N)
                )
        else:
            if M is None:
                print("please assign image size M and N.")
            # if 's_' in estimator_name:
            #     image_init = get_random_data(target[:,], target, M=M, N=N, N_image=target.shape[0], mode='func', seed=seed)
            if not ensemble:
                image_init = np.random.normal(0, 1, (target.shape[0], M, N))
            else:
                image_init = np.random.normal(0, 1, (N_ensemble, M, N))
    elif type(image_init) is str:
        if image_init == "random phase":
            image_init = get_random_data(target, seed=seed)  # gaussian field

    if J is None:
        J = int(np.log2(min(M, N))) - 1

    # define calculator and estimator function
    if "s" in estimator_name:
        st_calc = Scattering2d(
            M,
            N,
            J,
            L,
            device,
            wavelets,
            l_oversampling=l_oversampling,
            frequency_factor=frequency_factor,
        )
        if mode == "image":
            if "2fields" not in estimator_name:
                st_calc.add_ref(ref=target)
            else:
                if image_b is None:
                    print("should provide a valid image_b.")
                else:
                    st_calc.add_ref_ab(ref_a=target, ref_b=image_b)
            if ensemble:
                ref_P00_mean = st_calc.ref_scattering_cov["P00"].mean(0)[None, :, :]
                if "iso" in estimator_name:
                    ref_P00_mean = ref_P00_mean.mean(2)[:, :, None]
                st_calc.ref_scattering_cov["P00"] = ref_P00_mean
        if mode == "estimator":
            if image_ref is None:
                if target_full is None:
                    temp = target
                else:
                    temp = target_full
                if normalization == "P00":
                    if reference_P00 is None:
                        st_calc.add_synthesis_P00(
                            s_cov=temp, if_iso="iso" in estimator_name
                        )
                    else:
                        st_calc.add_synthesis_P00(P00=reference_P00)
                else:
                    st_calc.add_synthesis_P11(
                        temp, "iso" in estimator_name, C11_criteria
                    )
            else:
                st_calc.add_ref(ref=image_ref)

        if estimator_name == "s_mean_iso":
            func_s = lambda x: st_calc.scattering_coef(x)["for_synthesis_iso"]
        if estimator_name == "s_mean":
            func_s = lambda x: st_calc.scattering_coef(x)["for_synthesis"]
        if "s_cov_func" in estimator_name:

            def func_s(image):
                s_cov_set = st_calc.scattering_cov(
                    image,
                    use_ref=True,
                    if_large_batch=if_large_batch,
                    C11_criteria=C11_criteria,
                    normalization=normalization,
                    pseudo_coef=pseudo_coef,
                    remove_edge=remove_edge,
                )
                return s_cov_func(s_cov_set, s_cov_func_params)

        if estimator_name == "s_cov_iso_para_perp":

            def func_s(image):
                result = st_calc.scattering_cov(
                    image,
                    use_ref=True,
                    if_large_batch=if_large_batch,
                    C11_criteria=C11_criteria,
                    normalization=normalization,
                    pseudo_coef=pseudo_coef,
                    remove_edge=remove_edge,
                )
                index_type, j1, l1, j2, l2, j3, l3 = result["index_for_synthesis_iso"]
                select = (index_type < 3) + ((l2 == 0) + (l2 == L // 2)) * (
                    (l3 == 0) + (l3 == L // 2) + (l3 == -1)
                )
                return result["for_synthesis_iso"][:, select]

        if estimator_name == "s_cov_iso_iso":

            def func_s(image):
                result = st_calc.scattering_cov(
                    image,
                    use_ref=True,
                    if_large_batch=if_large_batch,
                    C11_criteria=C11_criteria,
                    normalization=normalization,
                    pseudo_coef=pseudo_coef,
                    remove_edge=remove_edge,
                )
                coef = result["for_synthesis_iso"]
                index_type, j1, l1, j2, l2, j3, l3 = result["index_for_synthesis_iso"]
                return torch.cat(
                    (
                        (coef[:, index_type < 3]),
                        (
                            coef[:, index_type == 3]
                            .reshape(-1, L)
                            .mean(-1)
                            .reshape(len(coef), -1)
                        ),
                        (
                            coef[:, index_type == 4]
                            .reshape(-1, L)
                            .mean(-1)
                            .reshape(len(coef), -1)
                        ),
                        (
                            coef[:, index_type == 5]
                            .reshape(-1, L, L)
                            .mean((-2, -1))
                            .reshape(len(coef), -1)
                        ),
                        (
                            coef[:, index_type == 6]
                            .reshape(-1, L, L)
                            .mean((-2, -1))
                            .reshape(len(coef), -1)
                        ),
                    ),
                    dim=-1,
                )

        if estimator_name == "s_cov_iso":
            func_s = lambda x: st_calc.scattering_cov(
                x,
                use_ref=True,
                if_large_batch=if_large_batch,
                C11_criteria=C11_criteria,
                normalization=normalization,
                pseudo_coef=pseudo_coef,
                remove_edge=remove_edge,
            )["for_synthesis_iso"]
        if estimator_name == "s_cov":
            func_s = lambda x: st_calc.scattering_cov(
                x,
                use_ref=True,
                if_large_batch=if_large_batch,
                C11_criteria=C11_criteria,
                normalization=normalization,
                pseudo_coef=pseudo_coef,
                remove_edge=remove_edge,
            )["for_synthesis"]
        if estimator_name == "s_cov_2fields_iso":

            def func_s(image):
                result = st_calc.scattering_cov_2fields(
                    image,
                    image_b,
                    use_ref=True,
                    if_large_batch=if_large_batch,
                    C11_criteria=C11_criteria,
                    normalization=normalization,
                    remove_edge=remove_edge,
                )
                select = (
                    (result["index_for_synthesis_iso"][0] != 1)
                    * (result["index_for_synthesis_iso"][0] != 3)
                    * (result["index_for_synthesis_iso"][0] != 7)
                    * (result["index_for_synthesis_iso"][0] != 11)
                    * (result["index_for_synthesis_iso"][0] != 15)
                    * (result["index_for_synthesis_iso"][0] != 19)
                )
                return result["for_synthesis_iso"][:, select]

        if estimator_name == "s_cov_2fields":

            def func_s(image):
                result = st_calc.scattering_cov_2fields(
                    image,
                    image_b,
                    use_ref=True,
                    if_large_batch=if_large_batch,
                    C11_criteria=C11_criteria,
                    normalization=normalization,
                    remove_edge=remove_edge,
                )
                select = (
                    (result["index_for_synthesis"][0] != 1)
                    * (result["index_for_synthesis"][0] != 3)
                    * (result["index_for_synthesis"][0] != 7)
                    * (result["index_for_synthesis"][0] != 11)
                    * (result["index_for_synthesis"][0] != 15)
                    * (result["index_for_synthesis"][0] != 19)
                )
                return result["for_synthesis"][:, select]

    if "alpha_cov" in estimator_name:
        aw_calc = AlphaScattering2d_cov(M, N, J, L, wavelets=wavelets, device=device)
        func_s = lambda x: aw_calc.forward(x)

    # power spectrum
    if ps_bins is None:
        ps_bins = J - 1

    def func_ps(image):
        ps, _ = get_power_spectrum(image, bins=ps_bins, bin_type=ps_bin_type)
        return torch.cat(
            (
                (image.mean((-2, -1)) / image.std((-2, -1)))[:, None],
                image.var((-2, -1))[:, None],
                ps,
            ),
            axis=-1,
        )

    # bispectrum
    if bi:
        if bispectrum_bins is None:
            bispectrum_bins = J - 1
        bi_calc = Bispectrum_Calculator(
            M, N, bins=bispectrum_bins, bin_type=bispectrum_bin_type, device=device
        )

        def func_bi(image):
            bi = bi_calc.forward(image)
            ps, _ = get_power_spectrum(
                image, bins=bispectrum_bins, bin_type=bispectrum_bin_type
            )
            return torch.cat(
                ((image.mean((-2, -1)) / image.std((-2, -1)))[:, None], ps, bi), axis=-1
            )

    def func(image):
        coef_list = []
        if estimator_name != "":
            coef_list.append(func_s(image))
        if ps:
            coef_list.append(func_ps(image))
        if bi:
            coef_list.append(func_bi(image))
        if phi4:
            coef_list.append(func_phi4(image))
        if phi4_j:
            coef_list.append(func_phi4_j(image, J))
        if hist:
            coef_list.append(hist_factor * func_hist(image))
        if hist_j:
            coef_list.append(hist_j_factor * func_hist_j(image, J))
        return torch.cat(coef_list, axis=-1)

    # define loss function
    def quadratic_loss(target, model):
        return ((target - model) ** 2).mean() * 1e8

    # synthesis
    kw_synthesis = dict(
        mode=mode,
        optim_algorithm=optim_algorithm,
        steps=steps,
        learning_rate=learning_rate,
        device=device,
        precision=precision,
        print_each_step=print_each_step,
        Fourier=Fourier,
        ensemble=ensemble,
        alpha_init=alpha_init,
    )
    if N_synthesis > 1:
        image_syn_list = []
        for k in range(N_synthesis):
            image_syn = synthesis_general(
                target,
                image_init,
                func,
                quadratic_loss,
                seed=None if seed is None else seed + k,
                **kw_synthesis,
            )
            image_syn_list.append(image_syn)
        return np.stack(image_syn_list)

    else:
        image_syn = synthesis_general(
            target,
            image_init,
            func,
            quadratic_loss,
            seed=seed,
            **kw_synthesis,
        )
        return image_syn


def synthesis_general(
    target,
    image_init,
    estimator_function,
    loss_function,
    mode="image",
    optim_algorithm="LBFGS",
    steps=100,
    learning_rate=0.5,
    device="gpu",
    precision="single",
    print_each_step=False,
    Fourier=False,
    ensemble=False,
    seed=None,
    alpha_init=0.01,
):
    # formating target and image_init (to tensor, to cuda)
    if type(target) is np.ndarray:
        target = torch.from_numpy(target)
    if type(image_init) is np.ndarray:
        image_init = torch.from_numpy(image_init)
    if precision == "double":
        target = target.type(torch.DoubleTensor)
        image_init = image_init.type(torch.DoubleTensor)
    else:
        target = target.type(torch.FloatTensor)
        image_init = image_init.type(torch.FloatTensor)
    if device == "gpu":
        target = target.cuda()
        image_init = image_init.cuda()

    # calculate statistics for target images
    if mode == "image":
        estimator_target = estimator_function(target)
    if mode == "estimator":
        estimator_target = target
    print("# of estimators: ", estimator_target.shape[-1])

    # define optimizable image model
    class OptimizableImage(torch.nn.Module):
        def __init__(self, base_image, delta_init, alpha=1.0, Fourier=False):
            super().__init__()
            self.base_image = base_image
            self.param = torch.nn.Parameter(delta_init)
            self.alpha = alpha
            self.Fourier = Fourier

        @property
        def image(self):
            if self.Fourier:
                delta = torch.fft.ifftn(
                    self.param[0] + 1j * self.param[1], dim=(-2, -1)
                ).real
            else:
                delta = self.param
            return self.base_image + self.alpha * delta

    if Fourier:
        temp = torch.fft.fftn(image_init, dim=(-2, -1))
        input_init = torch.cat((temp.real[None, ...], temp.imag[None, ...]), dim=0)
    else:
        input_init = image_init

    ### Optimizing delta relative to the backbone ###
    if seed is not None:
        torch.manual_seed(seed)
    delta_init = alpha_init * input_init.std() * torch.randn_like(input_init)
    image_model = OptimizableImage(input_init, delta_init, alpha=1.0, Fourier=Fourier)

    # define optimizer
    if optim_algorithm == "Adam":
        optimizer = torch.optim.Adam(image_model.parameters(), lr=learning_rate)
    elif optim_algorithm == "NAdam":
        optimizer = torch.optim.NAdam(image_model.parameters(), lr=learning_rate)
    elif optim_algorithm == "SGD":
        optimizer = torch.optim.SGD(image_model.parameters(), lr=learning_rate)
    elif optim_algorithm == "Adamax":
        optimizer = torch.optim.Adamax(image_model.parameters(), lr=learning_rate)
    elif optim_algorithm == "LBFGS":
        optimizer = torch.optim.LBFGS(
            image_model.parameters(),
            lr=learning_rate,
            max_iter=steps,
            max_eval=None,
            tolerance_grad=1e-19,
            tolerance_change=1e-19,
            history_size=min(steps // 2, 150),
            line_search_fn=None,
        )

    def closure():
        optimizer.zero_grad()
        loss = 0
        estimator_model = estimator_function(image_model.image)
        if ensemble:
            loss = loss_function(estimator_model.mean(0), estimator_target.mean(0))
        else:
            loss = loss_function(estimator_model, estimator_target)
        if print_each_step:
            if optim_algorithm == "LBFGS" or (
                optim_algorithm != "LBFGS" and (i % 100 == 0 or i % 100 == -1)
            ):
                if not ensemble:
                    print((estimator_model - estimator_target).abs().max())
                    print(
                        "max residual: ",
                        np.max(
                            (estimator_model - estimator_target)
                            .abs()
                            .detach()
                            .cpu()
                            .numpy()
                        ),
                        ", mean residual: ",
                        np.mean(
                            (estimator_model - estimator_target)
                            .abs()
                            .detach()
                            .cpu()
                            .numpy()
                        ),
                    )
                else:
                    print(
                        (estimator_model.mean(0) - estimator_target.mean(0)).abs().max()
                    )
                    print(
                        "max residual: ",
                        np.max(
                            (estimator_model.mean(0) - estimator_target.mean(0))
                            .abs()
                            .detach()
                            .cpu()
                            .numpy()
                        ),
                        ", mean residual: ",
                        np.mean(
                            (estimator_model.mean(0) - estimator_target.mean(0))
                            .abs()
                            .detach()
                            .cpu()
                            .numpy()
                        ),
                    )
        loss.backward()
        return loss

    # optimize
    t_start = time.time()
    if not ensemble:
        print(
            "max residual: ",
            np.max(
                (estimator_function(image_model.image) - estimator_target)
                .abs()
                .detach()
                .cpu()
                .numpy()
            ),
            ", mean residual: ",
            np.mean(
                (estimator_function(image_model.image) - estimator_target)
                .abs()
                .detach()
                .cpu()
                .numpy()
            ),
        )
    else:
        print(
            "max residual: ",
            np.max(
                (
                    estimator_function(image_model.image).mean(0)
                    - estimator_target.mean(0)
                )
                .abs()
                .detach()
                .cpu()
                .numpy()
            ),
            ", mean residual: ",
            np.mean(
                (
                    estimator_function(image_model.image).mean(0)
                    - estimator_target.mean(0)
                )
                .abs()
                .detach()
                .cpu()
                .numpy()
            ),
        )
    if optim_algorithm == "LBFGS":
        i = 0
        optimizer.step(closure)
    else:
        for i in range(steps):
            # print('step: ', i)
            optimizer.step(closure)
    if not ensemble:
        print(
            "max residual: ",
            np.max(
                (estimator_function(image_model.image) - estimator_target)
                .abs()
                .detach()
                .cpu()
                .numpy()
            ),
            ", mean residual: ",
            np.mean(
                (estimator_function(image_model.image) - estimator_target)
                .abs()
                .detach()
                .cpu()
                .numpy()
            ),
        )
    else:
        print(
            "max residual: ",
            np.max(
                (
                    estimator_function(image_model.image).mean(0)
                    - estimator_target.mean(0)
                )
                .abs()
                .detach()
                .cpu()
                .numpy()
            ),
            ", mean residual: ",
            np.mean(
                (
                    estimator_function(image_model.image).mean(0)
                    - estimator_target.mean(0)
                )
                .abs()
                .detach()
                .cpu()
                .numpy()
            ),
        )
    t_end = time.time()
    print("time used: ", t_end - t_start, "s")

    return image_model.image.cpu().detach().numpy()
