from pysb import *
import numpy as np
from scipy.constants import N_A
import re
from pysb.bng import generate_equations

Model()


Monomer('kenz1')
Monomer('kenz2')
Monomer('kenz3')
Monomer('kenz4')
Monomer('kenz5')


Monomer('AA')
Monomer('APC')

Monomer('Ca')
Monomer('CaPLC')
Monomer('CaPLCcomplex')
Monomer('CaPLA2_active')


Monomer('craf1')
Monomer('craf1_active')
Monomer('craf1_active_active')
Monomer('craf_deph')
Monomer('craf_active_active_deph')

Monomer('DAG')
Monomer('DAGPKCmemb_active')
Monomer('DAGCaPLA2_active')

Monomer('G_active_GTP')
Monomer('GqPLC')
Monomer('GqCaPLC')
Monomer('GqCaPLCcomplex')

Monomer('GTPRas')

Monomer('IP3')
Monomer('Inositiol')
Monomer('MAPK_active')
Monomer('MAPK_active_complex')
Monomer('MAPK_active_feedback')
Monomer('MAPKK')
Monomer('MAPKtyr')
Monomer('MAPK')
Monomer('MAPKKthr')
Monomer('MAPKK_active')
Monomer('MAPKKser')
Monomer('MAPKKdephser')
Monomer('MAPKKdeph')

Monomer('MKP1')
Monomer('MKP1thrdeph')

Monomer('PKCactraf')

Monomer('PPh2A')
Monomer('PIP2')
Monomer('PIP2CaPLA2_active')
Monomer('PIP2PLA2_active')

Monomer('PLC')
Monomer('PC')
Monomer('PKCi')
Monomer('PKCa')
Monomer('PKCbasal_active')
Monomer('AACaPKC_active')
Monomer('AAPKC_active')
Monomer('AADAGPKC')
Monomer('AADAGPKC_active')
Monomer('CaPKC')
Monomer('CaPKCi')
Monomer('CaPKCmemb_active')
Monomer('DAGPKC')
Monomer('DAGCaPKC')
Monomer('DAGCaPKCmemb_active')

Monomer('PLA2cyto')
Monomer('PLA2Ca_active')
Monomer('PLA2ctyo')
Monomer('PLA2_active')

Monomer('tempPIP2')

Monomer('RafGTPRas_active')
Monomer('RafGTPRas_active_1')
Monomer('RafGTPRas_active_2')

vol = 1.6667e-21  # uM
vol *= 10e6  # uM to M
scale_factor = N_A * vol

# DAG	0	0
# AA	0	0
# IP3	0	0
# Glu	0	0
# MAPK*	0	0
# BetaGamma	0	0
# G*GTP	0	0
# G*GDP	0	0
# PKA-active	0	0
# CaM-Ca4	0	0
# CaM-Ca3	0	0
# CaM-TR2-Ca2	0	0
# CaM(Ca)n-CaNAB	0	0
# PP2A	0.12	0
# CaNAB-Ca4	0	0
# cAMP	0	0
# Gs-alpha	0	0
# synapse	0	0
# CaMKII-act	0	0
# CaN-active	0	0
# tot_PP1-act	0	0
# PLA2-Ca*	0	0
# PIP2-PLA2*	0	0
# PIP2-Ca-PLA2*	0	0
# DAG-Ca-PLA2*	0	0
# PLA2*-Ca	0	0
# PLA2*	0	0
# Inositol	0	1
# PC	0	1
# PLC-Ca	0	0
# PLC-Ca-Gq	0	0
# PLC-Gq	0	0
# Rec-Glu	0	0
# Rec-Gq	0	0
# Rec-Glu-Gq	0	0
# mGluRAntag	0	1
# Blocked-rec-Gq	0	0
# craf-1*	0	0
# craf-1**	0	0
# MAPK-tyr	0	0
# MAPKK*	0	0
# MAPKK-ser	0	0
# Raf-GTP-Ras*	0	0
# GEF-Gprot-bg	0	0
# GEF*	0	0
# GTP-Ras	0	0
# CaMKII-CaM	0	0
# CaMKII-thr286*-CaM	0	0
# GAP*	0	0
# inact-GEF*	0	0
# CaM-GEF	0	0
# CaMKII***	0	0
# CaMKII-thr286	0	0
# CaMK-thr306	0	0
# CaNAB-Ca2	0	0
# CaMCa3-CaNAB	0	0
# CaMCa2-CANAB	0	0
# CaMCa4-CaNAB	0	0
# R2C2-cAMP	0	0
# R2C2-cAMP2	0	0
# R2C2-cAMP3	0	0
# R2C2-cAMP4	0	0
# R2C-cAMP4	0	0
# R2-cAMP4	0	0
# inhibited-PKA	0	0
# AC1-CaM	0	0
# AC2*	0	0
# AC2-Gs	0	0
# AC1-Gs	0	0
# AC2*-Gs	0	0
# cAMP-PDE*	0	0
# CaM.PDE1	0	0
# neurogranin*	0	0
# PP1-I1*	0	0
# PP1-I1	0	0


# temp-PIP2	2.5	1
# Ca	0.08	0
# PIP2	2.5	1
# MKP-1	0.0032	0
# PPhosphatase2A	0.224	0
# PP1-active	1.8	0

# PKC-active	2.1222e-16	0
# PKC-Ca	3.7208e-17	0
# PKC-DAG-AA*	4.9137e-18	0
# PKC-Ca-AA*	1.75e-16	0
# PKC-Ca-memb*	1.3896e-17	0
# PKC-DAG-memb*	9.4352e-21	0
# PKC-basal*	0.02	0
# PKC-AA*	1.8133e-17	0
# PKC-Ca-DAG	8.4632e-23	0
# PKC-DAG	1.161e-16	0
# PKC-DAG-AA	2.5188e-19	0
# PKC-cytosolic	1	0
# PLA2-cytosolic	0.4	0

# APC	30	1
# PLC	0.8	0
# G-GDP	1	0
# mGluR	0.3	0
# craf-1	0.2	0
# MAPKK	0.18	0
# MAPK	0.36	0
# inact-GEF	0.1	0
# GDP-Ras	0.2	0
# GAP	0.002	0
# neurogranin-CaM	0	0
# neurogranin	10	0
# I1	1.8	0
# I1*	0.001	0
# CaMKII	70	0
# total-CaMKII	70	1
# tot_CaM_CaMKII	0	0
# tot_autonomous_CaMKII	0	0
# CaM	20	0
# CaNAB	1	0
# R2C2	0.5	0
# PKA-inhibitor	0.25	0
# ATP	5000	1
# AC1	0.02	0
# AC2	0.015	0
# AMP	3.2549e+05	0
# cAMP-PDE	0.45	0
# PDE1	2	0


def initials_pkc():
    #Reactions for group /kinetics/PKC
    # PKC-Ca	3.7208e-17	0
    # PKC-DAG-AA*	4.9137e-18	0
    # PKC-Ca-AA*	1.75e-16	0
    # PKC-Ca-memb*	1.3896e-17	0
    # PKC-DAG-memb*	9.4352e-21	0
    # PKC-basal*	0.02	0
    # PKC-AA*	1.8133e-17	0
    # PKC-Ca-DAG	8.4632e-23	0
    # PKC-DAG	1.161e-16	0
    # PKC-DAG-AA	2.5188e-19	0
    # PKC-cytosolic	1	0

    Parameter("PKCCa_0",	3.7208e-17 * scale_factor)
    Initial(CaPKC(), PKCCa_0)
    Parameter("PKCDAG_AA_active_0",	4.9137e-18 * scale_factor)
    Initial(AADAGPKC_active(), PKCDAG_AA_active_0)
    Parameter("PKCCa_AA_active_0",	1.75e-16 * scale_factor)
    Initial(AACaPKC_active(), PKCCa_AA_active_0)
    Parameter("PKCCa_memb_active_0",	1.3896e-17 * scale_factor)
    Initial(CaPKCmemb_active(), PKCCa_memb_active_0)
    Parameter("PKCDAG_memb_active_0",	9.4352e-21 * scale_factor)
    Initial(DAGCaPKCmemb_active(), PKCDAG_memb_active_0)
    Parameter("PKCbasal_active_0",	0.02 * scale_factor)
    Initial(PKCbasal_active(), PKCbasal_active_0)

    Parameter("PKCAA_active_0",	1.8133e-17 * scale_factor)
    Initial(AAPKC_active(), PKCAA_active_0)

    Parameter("PKCCa_DAG_active_0",	8.4632e-23 * scale_factor)

    Parameter("PKCDAG_0",	1.161e-16*scale_factor)
    Initial(DAGCaPKC(), PKCDAG_0)
    Parameter("PKCDAG_AA_0",	2.5188e-19*scale_factor)
    Initial(AADAGPKC(), PKCDAG_AA_0)
    Parameter("PKCcyto_0", 1*scale_factor)
    Initial(PKCi(), PKCcyto_0)

def initials_dag():
    # DAG	150	1
    # Ca	1	1
    # AA	50	1
    # PKC-active	0.02	0
    Parameter('DAG_0', 150*scale_factor)
    Initial(DAG(), DAG_0)
    Parameter('Ca_0', 1*scale_factor)
    Initial(Ca(), Ca_0)
    Parameter('AA_0', 50*scale_factor)
    Initial(AA(), AA_0)
    Parameter('PKC_active_0', 0.02*scale_factor)
    Initial(PKCa(), PKC_active_0)


def forward_rate():
    Parameter('kf_p0133', 0.0133)
    Parameter('kf_p02', 0.02)
    Parameter('kf_p17', 0.17)
    Parameter('kf_p4', 0.4)
    Parameter('kf_1s', 1)
    Parameter('kf_2s', 2)
    Parameter('kf_100s', 100)
    Parameter('kf_600s', 600)
    Parameter('kf_8000s', 8000)
    Parameter('kf_18000s', 18000)
    Parameter('kf_1350000s', 1350000)
    Parameter('kf_1979996p93s', 197996.93)
    Parameter('kf_2519996p2s', 2519996.2)
    Parameter('kf_2529999s', 2529999)
    Parameter('kf_2760000s', 2760000)
    Parameter('kf_3000000s', 3000000)
    Parameter('kf_6000000s', 6000000)
    Parameter('kf_74999962p5s', 74999962.5)
    Parameter('kf_1950001p95s', 1950001.95)
    Parameter('kf_16199998p7s', 16199998.7)
    Parameter('kf_29999p85s', 29999.85)
    Parameter('kf_48000000s', 48000000)
    Parameter('kf_3299998p1', 3299998.1)
    Parameter('kf_24000000s', 24000000)
    Parameter('kf_60000s', 60000)
    Parameter('kf_1p2705s', 1.2705)
    Parameter('kf_1200s', 1200)
    Parameter('kf_1000000s', 1000000)
    Parameter('kf_12000s', 12000)
    Parameter('kf_9000000s', 9000000)
    Parameter('kf_3000', 3000)
    Parameter('kf_15000000', 15000000)
    Parameter('kf_3900003p9', 3900003.9)
    Parameter('kf_30000000s', 30000000)


def reverse_rates():
    Parameter('kb_0s', 0)
    Parameter('kb80s', 80)
    Parameter('kb_480s', 480)
    Parameter('kb_21p6', 21.6)
    Parameter('kb_p1', .1)
    Parameter('kb_50s', 50)
    Parameter('kb_192s', 192)
    Parameter('kb_1s', 1)
    Parameter('kb_40s', 40)
    Parameter('kb_4s', 4)
    Parameter('kb_p42', .42)
    Parameter('kb_p5', 5)
    Parameter('kb_16s', 16)
    Parameter('kb_p6s', .6)
    Parameter('kb_p5s', .5)
    Parameter('kb_144s', 144)
    Parameter('kb_240s', 240)
    Parameter('kb_p2s', .2)
    Parameter('kb_44p16', 44.16)
    Parameter('kb_25p00002s', 29.00002)
    Parameter('kb_3p5026s', 3.5026)
    Parameter('kb_8p6348s', 8.6348)
    Parameter('kb_p1s', 50)


def cat_rates():
    Parameter('kcat_4s', 4)
    Parameter('kcat_120s', 120)
    Parameter('kcat_20s', 20)
    Parameter('kcat_60s', 60)
    Parameter('kcat_36s', 36)
    Parameter('kcat_5p4s', 5.4)
    Parameter('kcat_48s', 48)
    Parameter('kcat_11p04', 11.04)
    Parameter('kcat_1s', 1)
    Parameter('kcat_p15s', .15)
    Parameter('kcat_p105', .105)
    Parameter('kcat_6s', 6)
    Parameter('kcat_10s', 10)


def all_rules():
    Rule('rule_1', PKCi() <> PKCbasal_active(), kf_1s, kb_50s)
    Rule('rule_2', AA() + PKCi() <> AAPKC_active(), kf_100s, kb_p1s)
    Rule('rule_3', CaPKC() <> CaPKCmemb_active(), kf_1p2705s, kb_3p5026s)
    Rule('rule_4', AA() + CaPKC() <> AACaPKC_active(), kf_1200s, kb_p1s)
    Rule('rule_5', DAGCaPKC() <> DAGPKCmemb_active(), kf_1s, kb_p1s)
    Rule('rule_6', AADAGPKC() <> AADAGPKC_active(), kf_2s, kb_p2s)
    Rule('rule_7', Ca() + PKCi() <> CaPKC(), kf_60000s, kb_p5s)
    Rule('rule_8', DAG() + CaPKCi() <> DAGCaPKC(), kf_8000s, kb_8p6348s)
    Rule('rule_9', DAG() + PKCi() <> DAGPKC(), kf_600s, kb_p1s)
    Rule('rule_10', AA() + DAGPKC() <> AADAGPKC(), kf_18000s, kb_p1s)

    Rule('rule_11_a', PKCa() +craf1() <> PKCactraf(), kf_29999p85s, kb_16s)
    Rule('rule_11_b', PKCactraf() >> PKCa() + craf1_active(), kcat_4s)
    Rule('rule_12_a', MAPK_active() + craf1_active() <> MAPK_active_feedback(), kf_1950001p95s, kb_40s )
    Rule('rule_12_b', MAPK_active_feedback() >> MAPK_active() + craf1_active_active(), kcat_10s )

    Rule('rule_13_a', PPh2A() + craf1_active() <> craf_deph(), kf_1979996p93s, kb_25p00002s )
    Rule('rule_13_b', craf_deph() >> PPh2A() + craf1(), kcat_6s)

    Rule('rule_14_a', PPh2A() + craf1_active_active() <> craf_active_active_deph(), kf_1979996p93s, kb_25p00002s )
    Rule('rule_14_b', craf_active_active_deph() >> PPh2A() + craf1_active(), kcat_6s)

    Rule('rule_15', craf1_active() + GTPRas() <> RafGTPRas_active(), kf_24000000s, kb_p5s)

    Rule('rule_16_a', RafGTPRas_active() + MAPKK() <> RafGTPRas_active_1(),  kf_3299998p1, kb_p42)
    Rule('rule_16_b', RafGTPRas_active_1() >> RafGTPRas_active() + MAPKKser(), kcat_p105)

    Rule('rule_17_a',  RafGTPRas_active() + MAPKKser() <> RafGTPRas_active_2(),  kf_3299998p1, kb_p42)
    Rule('rule_17_b', RafGTPRas_active_2() >> RafGTPRas_active() + MAPKK_active(), kb_p42 )

    Rule('rule_18_a', PPh2A() + MAPKKser() <> MAPKKdephser(), kf_1979996p93s, kb_25p00002s)
    Rule('rule_18_b', MAPKKdephser() >> PPh2A() + MAPKK(), kcat_6s)

    Rule('rule_19_a', PPh2A() + MAPKK_active() <> MAPKKdeph(), kf_1979996p93s, kb_25p00002s)
    Rule('rule_19_b', MAPKKdeph() >> PPh2A() + MAPKKser(), kcat_6s)

    Rule('rule_20_a', MAPKK_active() + MAPK() <> MAPKKthr(), kf_16199998p7s, kb_p6s)
    Rule('rule_20_b', MAPKKthr() >> MAPKK_active() + MAPKtyr(), kcat_p15s)

    Rule('rule_21_a', MAPKK_active() + MAPKtyr() <> MAPKKthr(), kf_16199998p7s, kb_p6s)
    Rule('rule_21_b', MAPKKthr() >> MAPKK_active() + MAPK_active(), kcat_p15s )

    Rule('rule_22_a', MKP1() + MAPKtyr() <> MKP1thrdeph(),  kf_74999962p5s, kb_4s)
    Rule('rule_22_b', MKP1thrdeph() >> MKP1() + MAPK(), kcat_1s)

    Rule('rule_23_a', MKP1() + MAPK_active() <> MKP1thrdeph(),  kf_74999962p5s, kb_4s)
    Rule('rule_23_b', MKP1thrdeph() >> MKP1() + MAPKtyr(), kcat_1s)
    Rule('rule_24', Ca() + PLC() <> CaPLC(), kf_3000000s, kb_1s)
    Rule('rule_25', G_active_GTP() + PLC() <> GqPLC(), kf_2529999s, kb_1s)
    Rule('rule_26', Ca() + GqPLC() <> GqCaPLC(), kf_3000000s, kb_1s)
    Rule('rule_27', G_active_GTP() + CaPLC() <> GqCaPLC(), kf_2529999s, kb_1s )
    Rule('rule_28', GqCaPLC() <> G_active_GTP() + CaPLC(), kf_p0133, kb_0s)

    Rule('rule_29_b', CaPLCcomplex() >> CaPLC() + DAG() + IP3(), kcat_10s)

    Rule('rule_30_b', GqCaPLCcomplex() >> GqCaPLC() + DAG() + IP3(), kcat_48s)
    Rule('rule_31', DAG() <> PC(), kf_p02, kb_0s)
    Rule('rule_32', IP3() <> Inositiol(), kf_1s, kb_0s)
    Rule('rule_35', Ca() + PLA2cyto() <>  PLA2Ca_active(), kf_1000000s, kb_p1)

    Rule('rule_36_b', kenz2()  >> PLA2Ca_active() + AA(), kcat_5p4s)


    Rule('rule_29_a', CaPLC() + PIP2() <> CaPLCcomplex(), kf_2519996p2s, kb_40s)
    Rule('rule_30_a',  GqCaPLC() + PIP2() <> GqCaPLCcomplex(),kf_48000000s, kb_192s)
    Rule('rule_37', tempPIP2() + PLA2Ca_active() <> PIP2CaPLA2_active(), kf_12000s, kb_p1s)
    Rule('rule_33', tempPIP2() + PLA2cyto() <> PIP2PLA2_active(), kf_1200s, kb_p5 )
    Rule('rule_34_a', PIP2PLA2_active() + PLA2cyto() <> kenz1(), kf_2760000s, kb_44p16)
    Rule('rule_34_b', kenz1() >> PIP2PLA2_active() + AA(), kcat_11p04)

    Rule('rule_36_a', PLA2Ca_active() + APC() <> kenz2(), kf_1350000s, kb_21p6)
    Rule('rule_38_a', PIP2CaPLA2_active() + APC() <> kenz3(), kf_9000000s, kb_144s)


    Rule('rule_38_b', kenz3() >> PIP2CaPLA2_active() + AA(), kcat_36s)

    Rule('rule_39', DAG() + PLA2Ca_active() <> DAGCaPLA2_active(), kf_3000, kb_4s)
    Rule('rule_40_a', DAGCaPLA2_active() + APC() <> kenz4(), kf_15000000, kb_240s)
    Rule('rule_40_b', kenz4() >> DAGCaPLA2_active() + AA(), kcat_60s)
    Rule('rule_41_a', MAPK_active() + PLA2cyto() <> MAPK_active_complex(),kf_3900003p9, kb80s )
    Rule('rule_41_b', MAPK_active_complex() >> MAPK_active() + PLA2_active(), kcat_20s)
    Rule('rule_42', PLA2_active() <> PLA2ctyo(), kf_p17, kb_0s)
    Rule('rule_43', Ca() + PLA2_active() <> CaPLA2_active(), kf_6000000s, kb_p1s)
    Rule('rule_44_a', CaPLA2_active() + APC() <> kenz5, kf_30000000s, kb_480s)
    Rule('rule_44_b', kenz5() >> CaPLA2_active() + AA() , kcat_120s)
    Rule('rule_45', AA() <> APC(), kf_p4, kb_p1s)


def constant_inits():
    Parameter("APC_0", 30e-6*scale_factor)
    Parameter("tempPIP2_0", 2.5e-6*scale_factor)
    Parameter('PIP2_0', 2.5e-6*scale_factor)
    Initial(APC(), APC_0)
    Initial(tempPIP2(), tempPIP2_0)
    Initial(PIP2(), PIP2_0)


def correct_rates():
    generate_equations(model)
    par_names = [p.name for p in model.parameters_rules()]
    rate_mask = np.array([p in model.parameters_rules() for p in model.parameters])
    param_values = np.array([p.value for p in model.parameters])
    param_values = np.repeat([param_values], 1, axis=0)
    rate_args = []
    par_vals = param_values[:,rate_mask]
    rate_order = []
    for rxn in model.reactions:
        rate_args.append([arg for arg in rxn['rate'].args if
                          not re.match("_*s", str(arg))])
        reactants = 0
        for i in rxn['reactants']:
            if not str(model.species[i]) == '__source()':
                reactants += 1
        rate_order.append(reactants)
    n_reactions = len(model.reactions)
    scaled_params = dict()
    for i in range(len(par_vals)):
        for j in range(n_reactions):
            rate = 1.0
            for r in rate_args[j]:
                if isinstance(r, Parameter):
                    rate *= par_vals[i][par_names.index(r.name)]
                elif isinstance(r, Expression):
                    raise ValueError('cupSODA does not currently support '
                                     'models with Expressions')
                else:
                    rate *= r
            # volume correction
            print(rate)
            rate *= (N_A *vol) ** (rate_order[j] - 1)
            scaled_params[r.name] = rate

    for i in scaled_params:
        print(i, scaled_params[i])
        model.parameters[i].value = scaled_params[i]


initials_pkc()
initials_dag()

forward_rate()
cat_rates()
reverse_rates()

constant_inits()

all_rules()
correct_rates()




if __name__ == "__main__":

    # for i in model.species:

    print(len(model.species))
    print(len(model.monomers))
    for i in model.reactions:
        print(i)
    for i in model.initial_conditions:
        print(i)
    # quit()
    from pysb.simulator.scipyode import ScipyOdeSimulator
    solver = ScipyOdeSimulator(model, tspan=np.linspace(0,20000,100))
    x = solver.run()
    import matplotlib.pyplot as plt
    plt.plot(np.array(x.species)[:,0])
    plt.show()