# -*- coding: utf-8 -*-

import numpy as np
from ..state import State
from copy import deepcopy

__all__ = ["TemperatureControl"]


def make_ladder(ndim, ntemps=None, Tmax=None):
    """
    Returns a ladder of :math:`\\beta \\equiv 1/T` under a geometric spacing that is determined by the
    arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:
    Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
    this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
    <http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
    ``ntemps`` is also specified.

    This function is originally from ``ptemcee`` `github.com/willvousden/ptemcee <https://github.com/willvousden/ptemcee>`_.

    Temperatures are chosen according to the following algorithm:
    * If neither ``ntemps`` nor ``Tmax`` is specified, raise an exception (insufficient
    information).
    * If ``ntemps`` is specified but not ``Tmax``, return a ladder spaced so that a Gaussian
    posterior would have a 25% temperature swap acceptance ratio.
    * If ``Tmax`` is specified but not ``ntemps``:
    * If ``Tmax = inf``, raise an exception (insufficient information).
    * Else, space chains geometrically as above (for 25% acceptance) until ``Tmax`` is reached.
    * If ``Tmax`` and ``ntemps`` are specified:
    * If ``Tmax = inf``, place one chain at ``inf`` and ``ntemps-1`` in a 25% geometric spacing.
    * Else, use the unique geometric spacing defined by ``ntemps`` and ``Tmax``.`

    Args:
        ndim (int): The number of dimensions in the parameter space.
        ntemps (int, optional): If set, the number of temperatures to generate.
        Tmax (float, optional): If set, the maximum temperature for the ladder.

    Returns:
        np.ndarray[ntemps]: Output inverse temperature (beta) array.

    Raises:
        ValueError: Improper inputs.

    """

    # make sure all inputs are okay
    if type(ndim) != int or ndim < 1:
        raise ValueError("Invalid number of dimensions specified.")
    if ntemps is None and Tmax is None:
        raise ValueError("Must specify one of ``ntemps`` and ``Tmax``.")
    if Tmax is not None and Tmax <= 1:
        raise ValueError("``Tmax`` must be greater than 1.")
    if ntemps is not None and (type(ntemps) != int or ntemps < 1):
        raise ValueError("Invalid number of temperatures specified.")

    # step size in temperature based on ndim
    tstep = np.array(
        [
            25.2741,
            7.0,
            4.47502,
            3.5236,
            3.0232,
            2.71225,
            2.49879,
            2.34226,
            2.22198,
            2.12628,
            2.04807,
            1.98276,
            1.92728,
            1.87946,
            1.83774,
            1.80096,
            1.76826,
            1.73895,
            1.7125,
            1.68849,
            1.66657,
            1.64647,
            1.62795,
            1.61083,
            1.59494,
            1.58014,
            1.56632,
            1.55338,
            1.54123,
            1.5298,
            1.51901,
            1.50881,
            1.49916,
            1.49,
            1.4813,
            1.47302,
            1.46512,
            1.45759,
            1.45039,
            1.4435,
            1.4369,
            1.43056,
            1.42448,
            1.41864,
            1.41302,
            1.40761,
            1.40239,
            1.39736,
            1.3925,
            1.38781,
            1.38327,
            1.37888,
            1.37463,
            1.37051,
            1.36652,
            1.36265,
            1.35889,
            1.35524,
            1.3517,
            1.34825,
            1.3449,
            1.34164,
            1.33847,
            1.33538,
            1.33236,
            1.32943,
            1.32656,
            1.32377,
            1.32104,
            1.31838,
            1.31578,
            1.31325,
            1.31076,
            1.30834,
            1.30596,
            1.30364,
            1.30137,
            1.29915,
            1.29697,
            1.29484,
            1.29275,
            1.29071,
            1.2887,
            1.28673,
            1.2848,
            1.28291,
            1.28106,
            1.27923,
            1.27745,
            1.27569,
            1.27397,
            1.27227,
            1.27061,
            1.26898,
            1.26737,
            1.26579,
            1.26424,
            1.26271,
            1.26121,
            1.25973,
        ]
    )

    if ndim > tstep.shape[0]:
        # An approximation to the temperature step at large
        # dimension
        tstep = 1.0 + 2.0 * np.sqrt(np.log(4.0)) / np.sqrt(ndim)
    else:
        # get correct step for dimension
        tstep = tstep[ndim - 1]

    # wheter to add the infinite temperature to the end
    appendInf = False
    if Tmax == np.inf:
        appendInf = True
        Tmax = None
        # non-infinite temperatures will now have 1 less
        ntemps = ntemps - 1

    if ntemps is not None:
        if Tmax is None:
            # Determine Tmax from ntemps.
            Tmax = tstep ** (ntemps - 1)
    else:
        if Tmax is None:
            raise ValueError(
                "Must specify at least one of ``ntemps" " and " "finite ``Tmax``."
            )

        # Determine ntemps from Tmax.
        ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

    betas = np.logspace(0, -np.log10(Tmax), ntemps)
    if appendInf:
        # Use a geometric spacing, but replace the top-most temperature with
        # infinity.
        betas = np.concatenate((betas, [0]))

    return betas


class TemperatureControl(object):
    """Controls the temperature ladder and operations in the sampler.

    All of the tempering features within Eryn are controlled from this class.
    This includes the evaluation of the tempered posterior, swapping between temperatures, and
    the adaptation of the temperatures over time. The adaptive tempering model can be
    found in the Eryn paper as well as the paper for `ptemcee`, which acted
    as a basis for the code below.

    Args:
        effective_ndim (int): Effective dimension used to determine temperatures if betas not given.
        nwalkers (int): Number of walkers in the sampler. Must maintain proper order of branches.
        ntemps (int, optional): Number of temperatures. If this is provided rather than ``betas``,
            :func:`make_ladder` will be used to generate the temperature ladder. (default: 1)
        betas (np.ndarray[ntemps], optional): If provided, will use as the array of inverse temperatures.
            (default: ``None``).
        Tmax (float, optional): If provided and ``betas`` is not provided, this will be included with
            ``ntemps`` when determing the temperature ladder with :func:`make_ladder`.
            See that functions docs for more information. (default: ``None``)
        adaptive (bool, optional): If ``True``, adapt the temperature ladder during sampling.
            (default: ``True``).
        adaptation_lag (int, optional): lag parameter from
            `arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>`_. ``adaptation_lag`` must be
            much greater than ``adapation_time``. (default: 10000)
        adaptation_time (int, optional): initial amplitude of adjustments from
            `arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>`_. ``adaptation_lag`` must be
            much greater than ``adapation_time``. (default: 100)
        stop_adaptation (int, optional): If ``stop_adaptation > 0``, the adapating will stop after
            ``stop_adaption`` steps. The number of steps is counted as the number times adaptation
            has happened which is generally once per sampler iteration. For example,
            if you only want to adapt temperatures during burn-in, you set ``stop_adaption = burn``.
            This can become complicated when using the repeating proposal options, so the
            user must be careful and verify constant temperatures in the backend.
            (default: -1)
        permute (bool, optional): If ``True``, permute the walkers in each temperature during
            swaps. (default: ``True``)
        skip_swap_supp_names (list, optional): List of strings that indicate supplemental keys that are not to be swapped.
            (default: ``[]``)
        non_adjacent_swaps (bool, optional): If ``True``, allow swaps between any pair of temperatures
            (not just adjacent). This can significantly improve mixing with many temperature levels.
            See Sambridge (2014) GJI 196:357-374. (default: ``False``)


    """

    def __init__(
        self,
        effective_ndim,
        nwalkers,
        ntemps=1,
        betas=None,
        Tmax=None,
        adaptive=True,
        adaptation_lag=10000,
        adaptation_time=100,
        stop_adaptation=-1,
        permute=True,
        skip_swap_supp_names=[],
        non_adjacent_swaps=False,
    ):

        if betas is None:
            if ntemps == 1:
                betas = np.array([1.0])
            else:
                # A compromise for building a temperature ladder for the case of rj.
                # We start by assuming that the dimensionality will be defined by the number of
                # components. We take that maximum divided by two, and multiply it with the higher
                # dimensional component.
                betas = make_ladder(effective_ndim, ntemps=ntemps, Tmax=Tmax)

        # store information
        self.nwalkers = nwalkers
        self.betas = betas
        self.ntemps = ntemps = len(betas)
        self.permute = permute
        self.skip_swap_supp_names = skip_swap_supp_names
        self.non_adjacent_swaps = non_adjacent_swaps

        # Running variance of log-likelihood per temperature (EWMA), for thermo-length spacing
        self.var_logl = np.zeros(self.ntemps, dtype=float)
        self._var_alpha = 0.05  # EWMA blending for variance updates
        self.thermo_adapt = False  # when True, use variance-based ladder adaptation

        # number of times adapted
        self.time = 0

        # store adapting inf
        self.adaptive = adaptive
        self.adaptation_time, self.adaptation_lag = adaptation_time, adaptation_lag
        self.stop_adaptation = stop_adaptation

        # For adjacent swaps: (ntemps-1,) array
        # For non-adjacent swaps: (ntemps, ntemps) matrix
        if self.non_adjacent_swaps:
            self.swaps_proposed = np.zeros((self.ntemps, self.ntemps))
            self.swaps_accepted_matrix = np.zeros((self.ntemps, self.ntemps))
            # Track adjacent pairs separately for proper statistics
            self.adj_swaps_proposed = np.zeros(self.ntemps - 1)
            self.adj_swaps_accepted = np.zeros(self.ntemps - 1)
        else:
            self.swaps_proposed = np.full(self.ntemps - 1, self.nwalkers)

    def compute_log_posterior_tempered(self, logl, logp, betas=None):
        """Compute the log of the tempered posterior

        Args:
            logl (np.ndarray): Log of the Likelihood. Can be 1D or 2D array. If 2D,
                must have shape ``(ntemps, nwalkers)``. If 1D, ``betas`` must be provided
                with the same shape.
            logp (np.ndarray): Log of the Prior. Can be 1D or 2D array. If 2D,
                must have shape ``(ntemps, nwalkers)``. If 1D, ``betas`` must be provided
                with the same shape.
            betas (np.ndarray[ntemps]): If provided, inverse temperatures as 1D array.
                If not provided, it will use ``self.betas``. (default: ``None``)

        Returns:
            np.ndarray: Log of the temperated posterior.

        Raises:
            AssertionError: Inputs are incorrectly shaped.

        """
        assert logl.shape == logp.shape
        tempered_logl = self.tempered_likelihood(logl, betas=betas)
        return tempered_logl + logp

    def tempered_likelihood(self, logl, betas=None):
        """Compute the log of the tempered Likelihood

        From `ptemcee`: "This is usually a mundane multiplication, except for the special case where
        beta == 0 *and* we're outside the likelihood support.
        Here, we find a singularity that demands more careful attention; we allow the
        likelihood to dominate the temperature, since wandering outside the
        likelihood support causes a discontinuity."

        Args:
            logl (np.ndarray): Log of the Likelihood. Can be 1D or 2D array. If 2D,
                must have shape ``(ntemps, nwalkers)``. If 1D, ``betas`` must be provided
                with the same shape.
            betas (np.ndarray[ntemps]): If provided, inverse temperatures as 1D array.
                If not provided, it will use ``self.betas``. (default: ``None``)

        Returns:
            np.ndarray: Log of the temperated Likelihood.

        Raises:
            ValueError: betas not provided if needed.

        """
        # perform calculation on 1D likelihoods.
        if logl.ndim == 1:
            if betas is None:
                raise ValueError(
                    "If inputing a 1D logl array, need to provide 1D betas array of the same length."
                )
            loglT = logl * betas

        else:
            if betas is None:
                betas = self.betas

            with np.errstate(invalid="ignore"):
                loglT = logl * betas[:, None]

        # anywhere the likelihood is nan, turn into -infinity
        loglT[np.isnan(loglT)] = -np.inf

        return loglT

    def do_swaps_indexing(
        self,
        i,
        iperm_sel,
        jperm_sel,
        dbeta,
        x,
        logP,
        logl,
        logp,
        inds=None,
        blobs=None,
        supps=None,
        branch_supps=None,
        j=None,
    ):

        # j defaults to i-1 for adjacent swaps (backward compatibility)
        if j is None:
            j = i - 1

        # for x and inds, just do full copy
        x_temp = {name: np.copy(x[name]) for name in x}
        if inds is not None:
            inds_temp = {name: np.copy(inds[name]) for name in inds}
        if branch_supps is not None:
            branch_supps_temp = {
                name: deepcopy(branch_supps[name]) for name in branch_supps
            }

        logl_temp = np.copy(logl[i, iperm_sel])
        logp_temp = np.copy(logp[i, iperm_sel])
        logP_temp = np.copy(logP[i, iperm_sel])
        if blobs is not None:
            blobs_temp = np.copy(blobs[i, iperm_sel])
        if supps is not None:
            supps_temp = deepcopy(supps[i, iperm_sel])

        # swap from j to i
        for name in x:
            # coords first
            x[name][i, iperm_sel, :, :] = x[name][j, jperm_sel, :, :]

            # then inds
            if inds is not None:
                inds[name][i, iperm_sel, :] = inds[name][j, jperm_sel, :]

            # do something special for branch_supps in case in contains a large amount of data
            # that is heavy to copy
            if branch_supps[name] is not None:
                tmp = branch_supps[name][j, jperm_sel, :]

                for key in self.skip_swap_supp_names:
                    tmp.pop(key)

                branch_supps[name][i, iperm_sel, :] = tmp
                """# where the inds are alive in the current permutation
                # need inds_temp because that is the original
                inds_i = np.where(inds_temp[name][i][iperm_sel])

                # gives the associated walker for each spot in the permuted array
                walker_inds_i = iperm_sel[inds_i[0]]

                # represents which permuted leaves are alive
                leaf_inds_i = inds_i[1]

                # all of these are at the same temperature
                temp_inds_i = np.full_like(leaf_inds_i, i)

                # repeat all for the i1 permutated temperature
                # need inds_temp because that is the original
                inds_i1 = np.where(inds_temp[name][i - 1][i1perm_sel])
                walker_inds_i1 = i1perm_sel[inds_i1[0]]
                leaf_inds_i1 = inds_i1[1]
                temp_inds_i1 = np.full_like(leaf_inds_i1, i - 1)

                # go through the values within each branch supplemental holder
                # do direct movement of things that need to change
                # rather than copying the whole thing
                for name2 in branch_supps[name].holder:
                    # store temperarily
                    bring_back_branch_supps = (
                        branch_supps[name]
                        .holder[name2][(temp_inds_i, walker_inds_i, leaf_inds_i)]
                        .copy()
                    )

                    # make switch from i1 to i
                    branch_supps[name].holder[name2][
                        (temp_inds_i, walker_inds_i, leaf_inds_i)
                    ] = branch_supps[name].holder[name2][
                        (temp_inds_i1, walker_inds_i1, leaf_inds_i1)
                    ]

                    # make switch from i to i1
                    branch_supps[name].holder[name2][
                        (temp_inds_i1, walker_inds_i1, leaf_inds_i1)
                    ] = bring_back_branch_supps"""

        # switch everything else from j to i
        logl[i, iperm_sel] = logl[j, jperm_sel]
        logp[i, iperm_sel] = logp[j, jperm_sel]
        logP[i, iperm_sel] = logP[j, jperm_sel] - dbeta * logl[j, jperm_sel]
        if blobs is not None:
            blobs[i, iperm_sel] = blobs[j, jperm_sel]
        if supps is not None:
            tmp_supps = supps[j, jperm_sel]
            for key in self.skip_swap_supp_names:
                tmp_supps.pop(key)
            supps[i, iperm_sel] = tmp_supps

        # switch x from i to j
        for name in x:
            x[name][j, jperm_sel, :, :] = x_temp[name][i, iperm_sel, :, :]
            if inds is not None:
                inds[name][j, jperm_sel, :] = inds_temp[name][i, iperm_sel, :]
            if branch_supps[name] is not None:
                tmp = branch_supps_temp[name][i, iperm_sel, :]

                for key in self.skip_swap_supp_names:
                    tmp.pop(key)
                branch_supps[name][j, jperm_sel, :] = tmp

        # switch the rest from i to j
        logl[j, jperm_sel] = logl_temp
        logp[j, jperm_sel] = logp_temp
        logP[j, jperm_sel] = logP_temp + dbeta * logl_temp

        if blobs is not None:
            blobs[j, jperm_sel] = blobs_temp
        if supps is not None:
            tmp_supps = supps_temp
            for key in self.skip_swap_supp_names:
                tmp_supps.pop(key)
            supps[j, jperm_sel] = tmp_supps

        return (x, logP, logl, logp, inds, blobs, supps, branch_supps)

    def temperature_swaps(
        self, x, logP, logl, logp, inds=None, blobs=None, supps=None, branch_supps=None
    ):
        """Perform parallel-tempering temperature swaps

        This function performs the swapping between temperatures. If ``non_adjacent_swaps=True``,
        swaps can occur between any pair of temperatures. Otherwise, it cascades from
        high temperature down to low temperature swapping only adjacent pairs.

        Args:
            x (dict): Dictionary with keys as branch names and values as coordinate arrays.
            logP (np.ndarray[ntemps, nwalkers]): Log of the posterior probability.
            logl (np.ndarray[ntemps, nwalkers]): Log of the Likelihood.
            logp (np.ndarray[ntemps, nwalkers]): Log of the prior probability.
            inds (dict, optional): Dictionary with keys as branch names and values as the index arrays
                indicating which leaves are used. (default: ``None``)
            blobs (object, optional): Blobs associated with each walker. (default: ``None``)
            supps (object, optional): :class:`eryn.state.BranchSupplemental` object. (default: ``None``)
            branch_supps (dict, optional): Dictionary with keys as branch names and values as
                :class:`eryn.state.BranchSupplemental` objects for each branch (can be ``None`` for some branches). (default: ``None``)

        Returns:
            tuple: All of the information that was input now swapped (output in the same order as input).

        """

        ntemps, nwalkers = self.ntemps, self.nwalkers

        # Update EWMA of per-temperature logl variance (energy fluctuations)
        cur_var = np.var(logl, axis=1)
        self.var_logl = (1.0 - self._var_alpha) * self.var_logl + self._var_alpha * cur_var

        # prepare information on how many swaps are accepted this time
        if self.non_adjacent_swaps:
            # For non-adjacent, track all pairs in a matrix, but also track adjacent for adaptation
            self.swaps_accepted_matrix_step = np.zeros((ntemps, ntemps))
            self.swaps_accepted = np.zeros(ntemps - 1)
        else:
            self.swaps_accepted = np.empty(ntemps - 1)

        if self.non_adjacent_swaps:
            # SAFE path: random disjoint pairs (uniform), size ~ ntemps/2
            order = np.random.permutation(ntemps)
            pair_iter = []
            k = 0
            while k + 1 < ntemps:
                i = order[k + 1]
                j = order[k]
                pair_iter.append((int(max(i, j)), int(min(i, j))))
                k += 2
        else:
            # Adjacent-only cascade from high to low: pairs are (i,j) with j=i-1
            pair_iter = [(ntemps - 1 - k, ntemps - 2 - k) for k in range(ntemps - 1)]

        # Attempt swaps for the chosen disjoint pairs
        for (i, j) in pair_iter:
            # get both temperature rungs
            bi = self.betas[i]
            bj = self.betas[j]
            dbeta = bj - bi

            # permute walker indices per temperature tier
            if self.permute:
                iperm = np.random.permutation(nwalkers)
                jperm = np.random.permutation(nwalkers)
            else:
                iperm = np.arange(nwalkers)
                jperm = np.arange(nwalkers)

            # Acceptance test per walker
            raccept = np.log(np.random.uniform(size=nwalkers))
            paccept = dbeta * (logl[i, iperm] - logl[j, jperm])
            sel = paccept > raccept
            num_accepted = np.sum(sel)

            # Accounting: proposed/accepted
            if self.non_adjacent_swaps:
                # Store in matrix at [i, j] where i > j (lower triangle, consistent with pair_iter)
                self.swaps_proposed[i, j] += nwalkers
                self.swaps_accepted_matrix[i, j] += num_accepted
                self.swaps_accepted_matrix_step[i, j] = num_accepted
                # Track adjacent pairs separately for proper statistics
                if abs(i - j) == 1:
                    adj_idx = min(i, j)
                    self.adj_swaps_proposed[adj_idx] += nwalkers
                    self.adj_swaps_accepted[adj_idx] += num_accepted
                    self.swaps_accepted[adj_idx] = num_accepted  # For this iteration only
            else:
                # adjacent mode keeps the original vector accounting; map (i,j) to swap index
                swap_idx = i  # since j=i-1 and i goes from ntemps-1..1
                self.swaps_accepted[swap_idx - 1] = num_accepted

            # Perform the actual swaps for selected walkers
            (x, logP, logl, logp, inds, blobs, supps, branch_supps) = self.do_swaps_indexing(
                i,
                iperm[sel],
                jperm[sel],
                dbeta,
                x,
                logP,
                logl,
                logp,
                inds=inds,
                blobs=blobs,
                supps=supps,
                branch_supps=branch_supps,
                j=j,
            )

        return (x, logP, logl, logp, inds, blobs, supps, branch_supps)

    def _get_ladder_adjustment(self, time, betas0, ratios):
        """
        Execute temperature adjustment according to dynamics outlined in
        `arXiv:1501.05823 <http://arxiv.org/abs/1501.05823>`_.
        """
        betas = betas0.copy()

        # Modulate temperature adjustments with a hyperbolic decay.
        decay = self.adaptation_lag / (time + self.adaptation_lag)
        kappa = decay / self.adaptation_time

        # Construct temperature adjustments.
        dSs = kappa * (ratios[:-1] - ratios[1:])

        # Compute new ladder (hottest and coldest chains don't move).
        deltaTs = np.diff(1 / betas[:-1])
        deltaTs *= np.exp(dSs)
        betas[1:-1] = 1 / (np.cumsum(deltaTs) + 1 / betas[0])

        # Don't mutate the ladder here; let the client code do that.
        return betas - betas0

    def adapt_temps(self):
        # determine ratios of swaps accepted to swaps proposed (the ladder is fixed)
        if self.non_adjacent_swaps:
            # For non-adjacent swaps, extract adjacent swap rates from matrices for ladder adaptation
            # Read from lower triangle [i+1, i] (since we store pairs as [max, min])
            adj_accepted = np.array([self.swaps_accepted_matrix[i+1, i] for i in range(self.ntemps-1)])
            adj_proposed = np.array([self.swaps_proposed[i+1, i] for i in range(self.ntemps-1)])
            # Avoid division by zero
            ratios = np.where(adj_proposed > 0, adj_accepted / adj_proposed, 0.25)
        else:
            ratios = self.swaps_accepted / self.swaps_proposed

        # Optional: thermodynamic-length spacing using running Var_beta[U]
        use_thermo = self.thermo_adapt and np.all(np.isfinite(self.var_logl)) and np.any(self.var_logl > 0)

        # adapt if desired
        if self.adaptive and self.ntemps > 1:
            if self.stop_adaptation < 0 or self.time < self.stop_adaptation:
                if use_thermo:
                    # Adjust neighbor gaps toward const / sqrt(var)
                    # IMPORTANT: Fix both endpoints (cold and hot chains don't move)
                    # Only adapt intermediate temperatures betas[1:-1]
                    if self.ntemps > 2:
                        # Current gaps (negative for descending betas)
                        current_gaps = np.diff(self.betas)

                        # Total span to preserve (negative)
                        total_span = np.sum(current_gaps)

                        # Variance weights (higher variance → larger gap magnitude)
                        w = 1.0 / np.sqrt(np.maximum(self.var_logl[:-1], 1e-12))
                        w /= np.sum(w)

                        # Desired gap sizes (maintain negative sign for descending)
                        desired_gaps = total_span * w

                        # Hyperbolic decay step size
                        decay = self.adaptation_lag / (self.time + self.adaptation_lag)
                        kappa = decay / self.adaptation_time

                        # Blend current and desired gaps
                        new_gaps = (1.0 - kappa) * current_gaps + kappa * desired_gaps

                        # Reconstruct ladder: fix cold, constrain hot within bounds
                        betas_new = np.empty_like(self.betas)
                        betas_new[0] = self.betas[0]  # Cold chain always at β=1
                        betas_new[1:] = betas_new[0] + np.cumsum(new_gaps)

                        # Constrain hot endpoint: prevent collapse AND allow reasonable adaptation
                        # Store initial beta at first adaptation step to set bounds
                        if not hasattr(self, 'beta_hot_initial'):
                            self.beta_hot_initial = self.betas[-1]

                        # Allow hot endpoint to adapt within ±50% of initial value
                        # This respects user's Tmax choice while preventing collapse to 0
                        beta_min = self.beta_hot_initial * 0.5  # Tmax can at most double
                        beta_max = self.beta_hot_initial * 1.5  # Tmax can at most halve
                        betas_new[-1] = np.clip(betas_new[-1], beta_min, beta_max)

                        self.betas = betas_new
                    # else: ntemps <= 2, no intermediate temperatures to adapt
                else:
                    dbetas = self._get_ladder_adjustment(self.time, self.betas, ratios)
                    self.betas += dbetas
            # only increase time if it is adaptive
            self.time += 1

    def temper_comps(self, state, adapt=True):
        """Perfrom temperature-related operations on a state.

        This includes making swaps and then adapting the temperatures for the next round.

        Args:
            state (object): Filled ``State`` object.
            adapt (bool, optional): If True, swaps are to be performed, but no
                adaptation is made. In this case, ``self.time`` does not increase by 1.
                (default: ``True``)

        Returns:
            :class:`eryn.state.State`: State object after swaps.

        """
        # get initial values
        logl = state.log_like
        logp = state.log_prior

        # do posterior just for the hell of it
        logP = self.compute_log_posterior_tempered(logl, logp)

        # make swaps
        x, logP, logl, logp, inds, blobs, supps, branch_supps = self.temperature_swaps(
            state.branches_coords,
            logP.copy(),
            logl.copy(),
            logp.copy(),
            inds=state.branches_inds,
            blobs=state.blobs,
            supps=state.supplemental,
            branch_supps=state.branches_supplemental,
        )

        if adapt and self.adaptive and self.ntemps > 1:
            self.adapt_temps()

        # create a new state out of the swapped information
        # TODO: make this more memory efficient?
        new_state = State(
            x,
            log_like=logl,
            log_prior=logp,
            blobs=blobs,
            inds=inds,
            betas=self.betas,
            supplemental=supps,
            branch_supplemental=branch_supps,
            random_state=state.random_state,
        )

        return new_state
