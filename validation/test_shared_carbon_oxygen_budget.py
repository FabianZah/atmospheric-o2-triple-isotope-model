"""Independent budget identities, stationary states and off-reference coupling."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from shared_carbon_oxygen_budget import SharedCarbonBudget, REFERENCE_PCO2_PPM
from updated_molecular_forward_model import UpdatedForwardInput
import updated_photosynthesis_transient as transient


@pytest.mark.parametrize('ppm',[50.,294.4,1000.,30000.])
@pytest.mark.parametrize('multiplier',[.94,1.])
def test_physical_carbon_mapping_and_initial_balance(ppm,multiplier):
    initial_bio=3.3e9
    c=SharedCarbonBudget(ppm,initial_bio,multiplier)
    ct,cs=c.initial_inventory
    assert c.pco2([ct,cs]) == pytest.approx(ppm,rel=1e-14)
    assert ct == pytest.approx(5.29e16*ppm/294.4,rel=1e-14)
    assert cs/ct == pytest.approx(.1)
    assert np.max(abs(c.tendency(c.initial_inventory,initial_bio))) < 4.
    assert c.constant_sources+c.balancing_flux-c.uptake_rate*ct-initial_bio == pytest.approx(0.,abs=4.)


@pytest.mark.parametrize('bio',[-2e16,0.,3e15])
def test_shared_biological_flux_cancels_pointwise(bio):
    c=SharedCarbonBudget(1000.,3.3e9)
    inventory=c.initial_inventory*np.array([1.4,.8])
    carbon_rate=c.tendency(inventory,bio)
    external=c.constant_sources+c.balancing_flux-c.uptake_rate*inventory[0]
    assert bio+sum(carbon_rate) == pytest.approx(external,rel=2e-14,abs=10.)


def test_reduced_carbon_solution_matches_independent_matrix_exponential():
    c=SharedCarbonBudget(1000.,0.)
    # For a known decaying respiration/photosynthesis imbalance, this is a
    # linear 3-state system (Ct anomaly, Cs anomaly, carbon source).
    tau=1731.
    matrix=np.array([[-c.uptake_rate-.1,1.,1.],[.1,-1.,0.],[0.,0.,-1/tau]])
    x0=np.array([0.,0.,6e15])
    scale=np.array([1e17,1e16,6e15])
    def rhs(t,z):
        ct,cs,source=z*scale
        return np.r_[c.tendency(c.initial_inventory+[ct,cs],-source),-source/tau]/scale
    times=np.array([0.,1.,10.,100.,1000.,22000.])
    solved=solve_ivp(rhs,(0.,22000.),x0/scale,method='Radau',t_eval=times,rtol=2e-10,atol=1e-12)
    exact=np.array([expm(matrix*t)@x0 for t in times])
    assert solved.success
    np.testing.assert_allclose(solved.y.T*scale,exact,rtol=2e-7,atol=2e6)


@pytest.mark.parametrize('name',['modern_reference','marine_endmember'])
def test_off_reference_coupling_and_no_step(name):
    initial=UpdatedForwardInput(.5,1000.,145.,name)
    request=transient.UpdatedPhotosynthesisTransientInput(initial,
        duration_years=22000.,sample_count=45,equilibrium_search_max_years=30000.)
    result=transient.run_updated_photosynthesis_transient(request)
    assert result.solver['initial_production_mol_per_year'] == pytest.approx(145e15/12.0107,rel=1e-12)
    assert result.solver['carbon_budget']['architecture']=='shared_carbon_oxygen_v1'
    for row in result.rows():
        assert row['pO2_pal']==row['carbon_driver_pO2_pal']
        assert row['pCO2_ppm']==pytest.approx(row['CO2_trop_mol']*294.4/5.29e16)
        assert min(row['CO2_trop_mol'],row['CO2_strat_mol'])>0
    unchanged=transient.run_updated_photosynthesis_transient(replace(request,photosynthesis_fraction=1.))
    assert max(abs(x-initial.p_co2_ppm) for x in unchanged.pco2_ppm)<1e-5
    assert max(abs(s.cap_delta17_prime_permil-unchanged.states[0].cap_delta17_prime_permil)
               for s in unchanged.states)<1e-6


@pytest.mark.parametrize('success,last',[ (False,500.),(True,500.) ])
def test_incomplete_shared_solution_is_rejected(success,last):
    fake=SimpleNamespace(success=success,sol=lambda t: np.ones(5),t=[0,last],message='injected')
    with patch.object(transient,'solve_ivp',return_value=fake):
        with pytest.raises(RuntimeError,match='did not complete'):
            transient.run_updated_photosynthesis_transient(
                transient.UpdatedPhotosynthesisTransientInput(UpdatedForwardInput()))


@pytest.mark.parametrize('po2,ppm',[(.1,50.),(2.,60000.)])
def test_stationary_domain_endpoints(po2,ppm):
    result=transient.run_updated_photosynthesis_transient(
        transient.UpdatedPhotosynthesisTransientInput(
            UpdatedForwardInput(po2,ppm,145.),photosynthesis_fraction=1.))
    np.testing.assert_allclose(result.pco2_ppm,ppm,rtol=1e-14)
    np.testing.assert_allclose(result.carbon_driver_po2_pal,po2,rtol=1e-14)


@pytest.mark.parametrize('po2,ppm,fraction',[(.15,1000.,.5),(1.,50.,1.25),(1.,60000.,.5)])
def test_outward_trajectory_rejected(po2,ppm,fraction):
    with pytest.raises(ValueError,match='leaves the supported'):
        transient.run_updated_photosynthesis_transient(
            transient.UpdatedPhotosynthesisTransientInput(
                UpdatedForwardInput(po2,ppm,145.),photosynthesis_fraction=fraction))


def test_low_production_equilibrium_is_not_finite_search_endpoint():
    request=transient.UpdatedPhotosynthesisTransientInput(
        UpdatedForwardInput(1.5,1000.,18.256264),duration_years=1000.,
        equilibrium_search_max_years=2000.)
    result=transient.run_updated_photosynthesis_transient(request)
    assert result.operational_equilibrium_time_years is None
    assert result.long_run_pco2_ppm == pytest.approx(1000.,abs=.001)
    assert result.long_run_state.o16o16/4.18e19 == pytest.approx(.75,abs=1e-5)
    assert result.solver['equilibrium_max_scaled_residual_per_year']<1e-12


def test_stationary_root_retries_success_with_unacceptable_residual():
    original=transient.root
    methods=[]
    def poor_first(fun,x0,**kwargs):
        methods.append(kwargs['method'])
        if kwargs['method']=='hybr':
            return SimpleNamespace(success=True,x=x0)
        return original(fun,x0,**kwargs)
    with patch.object(transient,'root',side_effect=poor_first):
        result=transient.run_updated_photosynthesis_transient(
            transient.UpdatedPhotosynthesisTransientInput(
                UpdatedForwardInput(1.5,1000.,18.256264),duration_years=1000.,
                equilibrium_search_max_years=2000.))
    assert methods==['hybr','lm']
    assert result.solver['equilibrium_root_methods']==methods
    assert result.solver['equilibrium_max_scaled_residual_per_year']<1e-12
    assert result.long_run_state.o16o16/4.18e19==pytest.approx(.75,abs=1e-5)


def test_stationary_root_fallback_still_rejects_bad_residual():
    def poor_both(fun,x0,**kwargs):
        return SimpleNamespace(success=True,x=x0)
    with patch.object(transient,'root',side_effect=poor_both):
        with pytest.raises(RuntimeError,match='equilibrium did not converge'):
            transient.run_updated_photosynthesis_transient(
                transient.UpdatedPhotosynthesisTransientInput(
                    UpdatedForwardInput(1.5,1000.,18.256264),duration_years=1000.,
                    equilibrium_search_max_years=2000.))


@pytest.mark.parametrize('po2,ppm,gpp',[(.5,1000.,145.),(1.,294.,290.),(1.5,10000.,72.5)])
def test_joint_rhs_uses_same_absolute_production_at_step(po2,ppm,gpp):
    # Inspect the implemented ODE itself, independently of its output aliases.
    class Captured(Exception):
        pass
    initial=UpdatedForwardInput(po2,ppm,gpp)
    def inspect(rhs,*args,**kwargs):
        rate=rhs(0.,np.ones(5))
        oxygen_rate=rate[0]*po2*4.18e19
        carbon_rate=sum(rate[3:]*SharedCarbonBudget(ppm,0.).initial_inventory)
        expected=-.5*gpp*1e15/12.0107
        assert oxygen_rate==pytest.approx(expected,rel=1e-10)
        assert carbon_rate==pytest.approx(-expected,rel=1e-10)
        assert abs(oxygen_rate+carbon_rate)/abs(expected)<1e-10
        raise Captured
    with patch.object(transient,'solve_ivp',side_effect=inspect),pytest.raises(Captured):
        transient.run_updated_photosynthesis_transient(transient.UpdatedPhotosynthesisTransientInput(initial))
