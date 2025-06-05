import matplotlib.pyplot as pp
import numpy as np
import sys
from copy import copy
import itertools as it
import networkx as nx


tstep = 0.1
t0 = 0
tmax = 1600
N = int((tmax-t0)/tstep)


if 'noC' in sys.argv: c_flag = False
else:
  from ctypes import c_long, c_int, c_double, POINTER
  from ctypes import CDLL
  evolib = CDLL('./lib_time_evo.so')

  evolib.time_evo.argtypes = (
         c_double,c_double,c_double,c_double,c_double,
         c_double,c_double,c_double,
         c_double,c_double,c_double,c_double,
         c_double,c_double,c_double,
         c_double,c_double,c_double,
         c_double,
         c_double,c_double,
         c_double,c_double,
         c_double,c_double,
         c_double,c_double,c_double,c_double,c_double,
         c_long,
         c_double,c_double,c_double,c_double,
         c_int,c_int,c_int,
         POINTER( c_double),POINTER( c_double),POINTER( c_double),POINTER( c_double),POINTER( c_double),POINTER( c_double),
         POINTER( c_double),POINTER( c_double),POINTER( c_double),POINTER( c_double),POINTER( c_double),
         POINTER( c_double),POINTER( c_double),POINTER( c_double),POINTER( c_double),POINTER( c_double))
  evolib.time_evo.restype = c_double
  c_flag = True



load_aac    = -0.1*np.load('norm-aac.npy')
load_ofc    = -0.1*np.load('norm-base.npy')
load_sst_up = -0.1*np.load('norm-sst-act.npy')
load_sst_dn = -0.1*np.load('norm-sst-inh.npy')
load_pv_up  = -0.1*np.load('norm-pv-act.npy')
load_pv_dn  = -0.1*np.load('norm-pv-inh.npy')

tgt_aac = load_aac
tgt_ofc = [load_ofc, load_sst_up, load_sst_dn, load_pv_up, load_pv_dn]



# activation function
f = lambda x, beta, th: 1.0 / (1.0 + np.exp(-beta * (x-th))) - 1.0/(1.0 + np.exp(beta*th))
# heaviside
H = lambda t: float(t >= 0.0)
# input
delta_trans = round(103/tstep) # ms, aac to ofc
delay_aac = 12 # ms, sound to aac, used in I() so no tstep
sound_dur = 20 # ms, used in I() so no tstep
I = lambda t: np.array([0.1*(H(t - (200+delay_aac)) - H(t - (200+sound_dur+delay_aac)) + H(t - (800+delay_aac)) - H(t - (800+sound_dur+delay_aac))),0,0,0,0]).T
IOGsu = lambda t, B, etype: 0.1*np.array([0,0,float( etype==1) *H(t-200)*(B+(10.0-10.0*B)/((t-190)+np.finfo(float).eps)),0,0]).T
IOGsd = lambda t, B, etype:-0.1*np.array([0,0,float( etype==2) *H(t-200)*(B+(10.0-10.0*B)/((t-190)+np.finfo(float).eps)),0,0]).T
IOGpu = lambda t, B, etype: 0.1*np.array([0,0,0,float( etype==3) *H(t-200)*(B+(10.0-10.0*B)/((t-190)+np.finfo(float).eps)),0]).T
IOGpd = lambda t, B, etype:-0.1*np.array([0,0,0,float( etype==4) *H(t-200)*(B+(10.0-10.0*B)/((t-190)+np.finfo(float).eps)),0]).T


def obj_fun( alpha_a, alpha_e, alpha_s, alpha_p, alpha_v,
             Waa, Wea, Wpa, # Wsa, Wva := 0
             Wee, Wse, Wpe, Wve, # Wae := 0
             Wes, Wps, Wvs, # Was, Wss := 0
             Wep, Wpp, Wvp, # Wap, Wsp := 0
             Wsv, # Wav, Wev, Wpv, Wvv := 0
             WsOGu, WsOGd,
             WpOGu, WpOGd,
             BsOG, BpOG,
             Aext=0, Eext=0, Sext=0, Pext=0, Vext=0,
             lfp_flag=False,beta=10.0, th=0.0, noise=False):

  Wsa = 0
  Wva = 0
  Wae = 0
  Was = 0
  Wss = 0
  Wap = 0
  Wsp = 0
  Wav = 0
  Wev = 0
  Wpv = 0
  Wvv = 0
  # all these by definition

  M = np.array([[ Waa, Wae, Was, Wap, Wav],
                [ Wea, Wee, Wes, Wep, Wev],
                [ Wsa, Wse, Wss, Wsp, Wsv],
                [ Wpa, Wpe, Wps, Wpp, Wpv],
                [ Wva, Wve, Wvs, Wvp, Wvv]])

  A = np.array( [ alpha_a, alpha_e, alpha_s, alpha_p, alpha_v]).T
  Iext = np.array( [0.0, 0.0, 0.0, 0.0, 0.0]).T

  val = 0
  for exp_type in range(5):
    # init
    Xp   = np.zeros((5,N))
    X    = np.zeros((5,N))
    Xif  = np.zeros((5,N))

    tt = t0
    for k in range(1, N):
      tt += tstep

      # time series
      X[:,k] = X[:,k-1] + tstep * Xp[:,k-1]

      km = k-1
      spec_km = max(0, k-1-delta_trans)
      Xp[:,k] = M @ X[:,km]
      Xp[1,k] -= M[1,0]*X[0,km]
      Xp[1,k] += M[1,0]*X[0,spec_km]
      Xp[3,k] -= M[3,0]*X[0,km]
      Xp[3,k] += M[3,0]*X[0,spec_km]


      Xif[:,k] = Xp[:,k] +Iext +WsOGu*IOGsu(tt-tstep,BsOG,exp_type) +WsOGd*IOGsd(tt-tstep,BpOG,exp_type) +WpOGu*IOGpu(tt-tstep,BsOG,exp_type) +WpOGd*IOGpd(tt-tstep,BpOG,exp_type) +I(tt-tstep)

      Xp[:,k] = A *( f( Xp[:,k] +Iext +I(tt-tstep) +WsOGu*IOGsu(tt-tstep,BsOG,exp_type) +WsOGd*IOGsd(tt-tstep,BpOG,exp_type) +WpOGu*IOGpu(tt-tstep,BsOG,exp_type) +WpOGd*IOGpd(tt-tstep,BpOG,exp_type), beta, th) -X[:,k-1] )

      if noise: X[1,k] += (np.random.uniform( 0.0, 1.0)-0.5)*0.5*10**-3

    # for k
    if 'corr' in sys.argv:
      if exp_type == 0: # only add aac for baseline experiment
        val += -1 * np.corrcoef(Xif[0,:], tgt_aac          )[0,1]
      val += -1 * np.corrcoef(Xif[1,:], tgt_ofc[exp_type])[0,1]
    elif 'dist' in sys.argv:
      if exp_type == 0: # only add aac for baseline experiment
        val += ((Xif[0,:]-tgt_aac)**2).sum()/10.0
      val += ((Xif[1,:]-tgt_ofc[exp_type])**2).sum()/10.0
    else: raise NotImplementedError

    if exp_type == 0:
      retX = X.copy()
      retXif = Xif.copy()
    else:
      retX = np.concatenate( [retX, X], axis=1)
      retXif = np.concatenate( [retXif, Xif], axis=1)

  return val,retX,retXif


### ENTRY POINT ###

lfp_flag = False

if   'corr' in sys.argv: obj_type = 0
elif 'dist' in sys.argv: obj_type = 1
else:
  sys.argv.append('dist')
  obj_type = 1 # hack, I know

def aux_f( arg_vec, ret_traj=False, isi_c=600.0, Imult=1.0, noise_c=False):

  if not c_flag:
    # changed by hand (for last figure only)
    global I
    I = lambda t: Imult*np.array([(H(t - (200+delay_aac)) - H(t - (200+sound_dur+delay_aac)) + H(t - (200+isi_c+delay_aac)) - H(t - (200+isi_c+sound_dur+delay_aac))),0,0,0,0]).T
    # calc
    temp = obj_fun( *arg_vec, lfp_flag=lfp_flag, noise=noise_c)
    # change back
    I = lambda t: np.array([(H(t - (200+delay_aac)) - H(t - (200+sound_dur+delay_aac)) + H(t - (800+delay_aac)) - H(t - (800+sound_dur+delay_aac))),0,0,0,0]).T
    if np.isnan( temp[0]): temp[0] = 9999999
    if not ret_traj: return temp[0]
    else:            return temp[0], temp[1], temp[2]

  else: # if c_flag
    temp_vec = list( map( c_double, arg_vec))
    if len(temp_vec) != 30:
      temp_vec.extend( list( map( c_double, [0.0]*5)) )
    ret_vec = np.zeros((5, 5*N))
    ret_vec_if = np.zeros((5, 5*N))
    temp = evolib.time_evo( *temp_vec,
            c_long(N), c_double(t0), c_double(tstep), c_double(isi_c), c_double(Imult),
            c_int( lfp_flag), c_int(obj_type), c_int(noise_c),
        tgt_aac.ctypes.data_as(POINTER(c_double)), tgt_ofc[0].ctypes.data_as(POINTER(c_double)),
        tgt_ofc[1].ctypes.data_as(POINTER(c_double)), tgt_ofc[2].ctypes.data_as(POINTER(c_double)),
        tgt_ofc[3].ctypes.data_as(POINTER(c_double)), tgt_ofc[4].ctypes.data_as(POINTER(c_double)),
        ret_vec[0].ctypes.data_as(POINTER(c_double)), ret_vec[1].ctypes.data_as(POINTER(c_double)),
        ret_vec[2].ctypes.data_as(POINTER(c_double)), ret_vec[3].ctypes.data_as(POINTER(c_double)),
        ret_vec[4].ctypes.data_as(POINTER(c_double)),
        ret_vec_if[0].ctypes.data_as(POINTER(c_double)), ret_vec_if[1].ctypes.data_as(POINTER(c_double)),
        ret_vec_if[2].ctypes.data_as(POINTER(c_double)), ret_vec_if[3].ctypes.data_as(POINTER(c_double)),
        ret_vec_if[4].ctypes.data_as(POINTER(c_double)) )
    if np.isnan( temp): temp = 9999999
    if not ret_traj: return temp
    else:            return temp, ret_vec, ret_vec_if



# solution
x0 = [
0.10300294760030633, 0.011102721119931531, 0.09408869738861059, 0.45826233303338915, 0.0006954301205058341,
0.3538845767507037, 0.3793656672581583, 0.4790899214329735,
1.163109787376575, 0.571500323740352, 0.9464482217968635, 1.844919843222439,
-1.7785148599962317, -2.79366548850593,   -0.796120455825248,
-1.1369116608266472, -0.9457529037879678, -2.2631859487527457,
-1.6943597444500715,
0.28, 0.185,
0.10, 0.22,
0.13, 0.93]


val,X,Xif = obj_fun( *x0, lfp_flag=lfp_flag)


e_names = ["Baseline:   ", "Op.Act. SST:", "Op.Inh. SST:", "Op.Act. PV: ", "Op.Inh. PV: "]
panelBratios = []
print( "SG ratios:")
for et in range(5):
  p1e = max( tgt_ofc[et][round(300/tstep):round(949/tstep)])
  p2e = max( tgt_ofc[et][round(950/tstep):])
  p1s = max( Xif[1, round(et*N + 300/tstep):round(et*N + 949/tstep)]) - Xif[1,round(100/tstep)]
  p2s = max( Xif[1, round(et*N + 950/tstep):round((et+1)*N)]) - Xif[1,round(100/tstep)]
  print( "-- {} p1e: {:.2f} p1s: {:.2f} p2e: {:.2f} p2s: {:.2f} --- \033[1mre: {:.2f} rs: {:.2f}\033[0m  ({:+.2f}, x{:.2f})".format( e_names[et], p1e, p1s, p2e, p2s, p2e/p1e, p2s/p1s, p2s/p1s - p2e/p1e, (p2s/p1s)/(p2e/p1e) ))
  panelBratios.append( [p2e/p1e, p2s/p1s] )
print( val)
print( aux_f(x0))
print( sys.argv)
if 'dist' in sys.argv: print('Within-sample MSE: ', (10.0*val)/(6*N))



fakeI = np.concatenate( [np.array( list( map( lambda x: I(x)[0], np.arange(t0, tmax, tstep))))]*5 )
fakeOGs = np.concatenate([np.array(list(map( lambda x: IOGsu(x, x0[23], et)[2] + IOGsd(x, x0[24], et)[2], np.arange(t0, tmax, tstep)))) for et in range(5)])
fakeOGp = np.concatenate([np.array(list(map( lambda x: IOGpu(x, x0[23], et)[3] + IOGpd(x, x0[24], et)[3], np.arange(t0, tmax, tstep)))) for et in range(5)])
fakeT = np.arange(t0, 5*tmax, tstep)
fakeOFC = np.concatenate( tgt_ofc)
fakeAAC = tgt_aac

pp.subplot(611)
pp.grid()
pp.plot( fakeI, 'g')
pp.plot( fakeOGs, 'b')
pp.plot( fakeOGp, 'r')
pp.legend(['Sound input', 'Opto SST', 'Opto PV'])
pp.axvline( tmax/tstep, color='k')
pp.axvline( 2*tmax/tstep, color='k')
pp.axvline( 3*tmax/tstep, color='k')
pp.axvline( 4*tmax/tstep, color='k')

pp.subplot(612)
pp.grid()
pp.plot( fakeT, Xif[0,:], 'g')
pp.plot( fakeT, X[0,:], 'g--')
pp.plot( np.arange(t0, tmax, tstep), fakeAAC, 'k')
pp.legend(['AAC Inp.', 'AAC Outp.', 'Negative AAC LFP'])
pp.axvline( tmax, color='k')
pp.axvline( 2*tmax, color='k')
pp.axvline( 3*tmax, color='k')
pp.axvline( 4*tmax, color='k')

pp.subplot(613)
pp.grid()
pp.plot( fakeT, Xif[1,:], 'purple')
pp.plot( fakeT, X[1,:], 'purple', linestyle='--')
pp.plot( fakeT, fakeOFC, 'k')
pp.legend(['Pyr. Inp.', 'Pyr. Outp.', 'Negative OFC LFP'])
pp.axvline( tmax, color='k')
pp.axvline( 2*tmax, color='k')
pp.axvline( 3*tmax, color='k')
pp.axvline( 4*tmax, color='k')

pp.subplot(614)
pp.grid()
pp.plot( fakeT, Xif[2,:], 'b')
pp.plot( fakeT, X[2,:], 'b--')
pp.legend(['SST Inp.', 'SST Outp.'])
pp.axvline( tmax, color='k')
pp.axvline( 2*tmax, color='k')
pp.axvline( 3*tmax, color='k')
pp.axvline( 4*tmax, color='k')

pp.subplot(615)
pp.grid()
pp.plot( fakeT, Xif[3,:], 'r')
pp.plot( fakeT, X[3,:], 'r--')
pp.legend(['PV Inp.', 'PV Outp.'])
pp.axvline( tmax, color='k')
pp.axvline( 2*tmax, color='k')
pp.axvline( 3*tmax, color='k')
pp.axvline( 4*tmax, color='k')

pp.subplot(616)
pp.grid()
pp.plot( fakeT, Xif[4,:],'brown')
pp.plot( fakeT, X[4,:],'brown', linestyle='--')
pp.legend(['VIP Inp.', 'VIP Outp.'])
pp.xlabel('Time (ms)')
pp.axvline( tmax, color='k')
pp.axvline( 2*tmax, color='k')
pp.axvline( 3*tmax, color='k')
pp.axvline( 4*tmax, color='k')

pp.get_current_fig_manager().window.showMaximized()
def on_resize(event):
  fig = pp.gcf()
  fig.tight_layout()
  fig.canvas.draw()
pp.gcf().canvas.mpl_connect('resize_event', on_resize)
pp.show()


# PANEL B; resize to about half a screen and apply tight layout, hspace=0.3

fakeTsingle = np.arange(t0, tmax, tstep)
labels=['BL', 'SST Act.', 'SST Inh.', 'PV Act.', 'PV Inh.']

for k in range(5):
  pp.subplot(5,1,k+1)
  pp.grid()

  pp.plot( fakeTsingle, Xif[1,k*N:(k+1)*N], 'purple')
  pp.plot( fakeTsingle, tgt_ofc[k], 'k')

  pp.ylim([(k==4)*-0.1-0.1,(k==4)*0.18+0.13])

  pp.annotate('', xy=(200, 0), xytext=(200, (k==4)*-0.08-0.08), arrowprops=dict(color='blue', headlength=3, headwidth=3, width=1) )
  pp.annotate('', xy=(800, 0), xytext=(800, (k==4)*-0.08-0.08), arrowprops=dict(color='blue', headlength=3, headwidth=3, width=1) )

  text = pp.text( 1000, (k==4)*0.11+0.09, labels[k] + ': {:.2f}'.format( panelBratios[k][0]), color='k')
  text = pp.annotate( '/', xycoords=text, xy=(1, 0), color='k')
  pp.annotate('{:.2f}'.format( panelBratios[k][1]), xycoords=text, xy=(1, 0), color='purple')

pp.xlabel('Time (ms)')
pp.show()





# minimization block
if ('local' in sys.argv) or ('global' in sys.argv):
  from scipy.optimize import minimize, OptimizeResult, differential_evolution, basinhopping

  iter_N = 0
  iter_N2 = 0
  def minimize_callback( intermediate_result: OptimizeResult):
    global iter_N
    iter_N += 1
    print("Iteration: ", iter_N)
    print(" -- current fun: ", intermediate_result.fun)

  def minimize_callback2( x, fun, accept):
    global iter_N2
    global dumpf
    iter_N2 += 1
    print("GLOBAL Iteration: ", iter_N2)
    print(" -- current fun: ", fun)
    dumpf.write( '\niter: ' + str(iter_N2) + '; fun = ' + str(fun) + '\n')
    dumpf.write( 'x0 = \n')
    dumpf.write( str(x))
    dumpf.write('\n')
    dumpf.flush()

  assert len(x0) == 25
  opt_bounds = [ (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), # alpha
          (-10, 10), (0, 20), (0, 20),# a excites a, e, p
          (0, 10), (0, 10), (0, 10), (0, 10), # e excites e, s, p, v
          (-20, 0), (-20, 0), (-20, 0), # s inhibits e, p, v
          (-20, 0), (-20, 0), (-10, 0), # p inhibits e, p, v
          (-10, 0), # v inhibits s
          (0, 2),(0, 2),
          (0, 2),(0, 2),     # bounds on opto strength
          (0.1, 1),
          (0.1, 1)        # bounds on opto decay
          ]

  if 'local' in sys.argv:
    print("starting local minimization")
    result = minimize( aux_f, x0, method='L-BFGS-B', bounds=opt_bounds,
          callback=minimize_callback)
  elif 'global' in sys.argv:
    print("starting global minimization")
    dumpf = open("min_dump-"+sys.argv[-1]+".txt", 'w')
    #result = differential_evolution( aux_f, opt_bounds,
    #      x0=x0, recombination=0.3, mutation=(0.7, 1.3), maxiter=100000,
    #      callback=minimize_callback)
    result = basinhopping( aux_f, x0, T=0.2, niter=100, stepsize=np.mean(x0),
            minimizer_kwargs={'bounds':opt_bounds, 'callback':minimize_callback},
          callback=minimize_callback2)
    dumpf.close()
  else: raise NotImplementedError

  print( result)
  print("x0 = [")
  for k in range(len(result.x)-1): print( str(result.x[k]) + ", ")
  print( str(result.x[-1]) + "]")
  print(sys.argv)

  # update x0
  x0 = list(map(float, result.x))

  val,X,Xif = obj_fun( *result.x, lfp_flag=lfp_flag)


  print( "SG ratios:")
  for et in range(5):
    p1e = max( tgt_ofc[et][round(300/tstep):round(949/tstep)])
    p2e = max( tgt_ofc[et][round(950/tstep):])
    p1s = max( Xif[1, round(et*N + 300/tstep):round(et*N + 949/tstep)]) - Xif[1,round(100/tstep)]
    p2s = max( Xif[1, round(et*N + 950/tstep):round((et+1)*N)]) - Xif[1,round(100/tstep)]
    print( "-- {} p1e: {:.2f} p1s: {:.2f} p2e: {:.2f} p2s: {:.2f} --- \033[1mre: {:.2f} rs: {:.2f}\033[0m  ({:+.2f}, x{:.2f})".format( e_names[et], p1e, p1s, p2e, p2s, p2e/p1e, p2s/p1s, p2s/p1s - p2e/p1e, (p2s/p1s)/(p2e/p1e) ))
  print( val)
  fakeOGs = np.concatenate([np.array(list(map( lambda x: IOGsu(x, x0[23], et)[2] + IOGsd(x, x0[24], et)[2], np.arange(t0, tmax, tstep)))) for et in range(5)])
  fakeOGp = np.concatenate([np.array(list(map( lambda x: IOGpu(x, x0[23], et)[3] + IOGpd(x, x0[24], et)[3], np.arange(t0, tmax, tstep)))) for et in range(5)])

  pp.subplot(611)
  pp.grid()
  pp.plot( fakeI, 'g')
  pp.plot( fakeOGs, 'b')
  pp.plot( fakeOGp, 'r')
  pp.legend(['Sound input', 'Opto SST', 'Opto PV'])
  pp.axvline( tmax/tstep, color='k')
  pp.axvline( 2*tmax/tstep, color='k')
  pp.axvline( 3*tmax/tstep, color='k')
  pp.axvline( 4*tmax/tstep, color='k')

  pp.subplot(612)
  pp.grid()
  pp.plot( fakeT, Xif[0,:], 'g')
  pp.plot( fakeT, X[0,:], 'g--')
  pp.plot( np.arange(t0, tmax, tstep), fakeAAC, 'k')
  pp.legend(['AAC Inp.', 'AAC Outp.', 'Negative AAC LFP'])
  pp.axvline( tmax, color='k')
  pp.axvline( 2*tmax, color='k')
  pp.axvline( 3*tmax, color='k')
  pp.axvline( 4*tmax, color='k')

  pp.subplot(613)
  pp.grid()
  pp.plot( fakeT, Xif[1,:], 'purple')
  pp.plot( fakeT, X[1,:], 'purple', linestyle='--')
  pp.plot( fakeT, fakeOFC, 'k')
  pp.legend(['Pyr. Inp.', 'Pyr. Outp.', 'Negative OFC LFP'])
  pp.axvline( tmax, color='k')
  pp.axvline( 2*tmax, color='k')
  pp.axvline( 3*tmax, color='k')
  pp.axvline( 4*tmax, color='k')

  pp.subplot(614)
  pp.grid()
  pp.plot( fakeT, Xif[2,:], 'b')
  pp.plot( fakeT, X[2,:], 'b--')
  pp.legend(['SST Inp.', 'SST Outp.'])
  pp.axvline( tmax, color='k')
  pp.axvline( 2*tmax, color='k')
  pp.axvline( 3*tmax, color='k')
  pp.axvline( 4*tmax, color='k')

  pp.subplot(615)
  pp.grid()
  pp.plot( fakeT, Xif[3,:], 'r')
  pp.plot( fakeT, X[3,:], 'r--')
  pp.legend(['PV Inp.', 'PV Outp.'])
  pp.axvline( tmax, color='k')
  pp.axvline( 2*tmax, color='k')
  pp.axvline( 3*tmax, color='k')
  pp.axvline( 4*tmax, color='k')

  pp.subplot(616)
  pp.grid()
  pp.plot( fakeT, Xif[4,:], 'brown')
  pp.plot( fakeT, X[4,:], 'brown', linestyle='--')
  pp.legend(['VIP Inp.', 'VIP Outp.'])
  pp.xlabel('Time (ms)')
  pp.axvline( tmax, color='k')
  pp.axvline( 2*tmax, color='k')
  pp.axvline( 3*tmax, color='k')
  pp.axvline( 4*tmax, color='k')

  pp.get_current_fig_manager().window.showMaximized()
  pp.gcf().canvas.mpl_connect('resize_event', on_resize)
  pp.show()
# if minimize


# PANEL A; resize to about half a screen and apply tight layout

connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]

G2 = nx.MultiDiGraph()

G2.add_nodes_from([0,1,2,3,4])

A = np.array([ x0[0], x0[1], x0[2], x0[3], x0[4]])

M = np.array([[ x0[5], 0.0,    0.0,    0.0,    0.0],
              [ x0[6], x0[8],  x0[12], x0[15], 0.0],
              [ 0.0,   x0[9],  0.0,    0.0,    x0[18]],
              [ x0[7], x0[10], x0[13], x0[16], 0.0],
              [ 0.0,   x0[11], x0[14], x0[17], 0.0]])

lb = []
for i in range(5):
  for j in range(5):
    lb.append( (i,j,M.T[i,j]) )

G2.add_weighted_edges_from(lb)

labels = {}
for item in lb: labels[(item[0],item[1])] = "{:.2f}".format( item[2])
indexes = [ v != '0.00' for v in labels.values()]

el = []
i = 0
for k in G2.edges():
  if indexes[i]: el.append( k)
  i += 1

labels2 = {}
for i in range(len(lb)):
  if indexes[i]: labels2[ ( lb[i][0], lb[i][1])] = "\n{:+.2f}".format(  lb[i][2])

pos = {0: np.array([ 0, 0.1]),
       1: np.array([ 0.15, 0.5]),
       2: np.array([ 0.65, 0.5]),
       3: np.array([ 0.85, 0]),
       4: np.array([ 0.6, 1.0])}


colors = ['green','purple', 'blue', 'red', 'brown']
nx.draw_networkx_nodes(G2, pos, node_color=colors)
struct_names = {0:'0: ACC', 1:'1: PYR', 2:'2: SST', 3:'3: PV', 4:'4: VIP'}
nd_labels = {k: "\n\n\n\n" + struct_names[k] + "\n$\\tau = {:.1f}$ ms".format( 1.0/A[k]) for k in range(5)}
nx.draw_networkx_labels(G2, pos, labels=nd_labels, font_size=13)
nx.draw_networkx_edges(G2, pos, edgelist=el, edge_color="grey", connectionstyle=connectionstyle)
nx.draw_networkx_edge_labels(G2, pos, labels2, connectionstyle=connectionstyle, label_pos=0.6,
                               font_color='blue', font_size=13, bbox={'alpha':0})
pp.show()




# robustness
mult = [0.95, 1.05]
rat_list = [[],[],[],[],[]]
#pp.subplot(211)
pp.plot( fakeT, fakeOFC, 'k')
for k in range(len(x0)):
  for j in range(2):
    new_x0 = copy( x0)
    new_x0[k] *= mult[j]

    val,X,Xif = aux_f( new_x0, ret_traj=True)

    pp.plot( fakeT, Xif[1,:], 'purple', alpha=0.3)

    for et in range(5):
      p1s = max( Xif[1, round(et*N + 300/tstep):round(et*N + 949/tstep)]) - Xif[1,round(100/tstep)]
      p2s = max( Xif[1, round(et*N + 950/tstep):round((et+1)*N)]) - Xif[1,round(100/tstep)]
      rat_list[et].append( p2s/p1s)
pp.axvline( tmax, color='k')
pp.axvline( 2*tmax, color='k')
pp.axvline( 3*tmax, color='k')
pp.axvline( 4*tmax, color='k')
pp.grid()
pp.xlabel('Time (ms)')
pp.title( 'Effect of perturbing parameters $\pm5\%$')

pp.show()
#pp.subplot(212)
def rand_jitter(arr):
  stdev = 0.07
  return arr + np.random.randn(len(arr)) * stdev

def jitter(ax, x, y, s=20, c='b', marker='o',
        cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, hold=None, **kwargs):
  return ax.scatter( rand_jitter(x), y, s=s, c=c, marker=marker, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, alpha=alpha, linewidths=linewidths, **kwargs)

fig, (ax1, ax2, ax3) = pp.subplots(3,1,sharex=True, height_ratios=[1,5,1])
fig.subplots_adjust(hspace=0.05)
labels=['BL', 'SST Act.', 'SST Inh.', 'PV Act.', 'PV Inh.']
bplot = ax2.boxplot( rat_list, False, '', patch_artist=True)
for patch in bplot['boxes']:
  patch.set_facecolor( 'purple')
  patch.set_alpha(0.5)
jitter(ax1, [1]*len(rat_list[0]), rat_list[0], c='k', alpha=0.6)
jitter(ax1, [2]*len(rat_list[0]), rat_list[1], c='k', alpha=0.6)
jitter(ax1, [3]*len(rat_list[1]), rat_list[2], c='k', alpha=0.6)
jitter(ax1, [4]*len(rat_list[1]), rat_list[3], c='k', alpha=0.6)
jitter(ax1, [5]*len(rat_list[2]), rat_list[4], c='k', alpha=0.6)
jitter(ax2, [1]*len(rat_list[0]), rat_list[0], c='k', alpha=0.6)
jitter(ax2, [2]*len(rat_list[0]), rat_list[1], c='k', alpha=0.6)
jitter(ax2, [3]*len(rat_list[1]), rat_list[2], c='k', alpha=0.6)
jitter(ax2, [4]*len(rat_list[1]), rat_list[3], c='k', alpha=0.6)
jitter(ax2, [5]*len(rat_list[2]), rat_list[4], c='k', alpha=0.6)
jitter(ax3, [1]*len(rat_list[0]), rat_list[0], c='k', alpha=0.6)
jitter(ax3, [2]*len(rat_list[0]), rat_list[1], c='k', alpha=0.6)
jitter(ax3, [3]*len(rat_list[1]), rat_list[2], c='k', alpha=0.6)
jitter(ax3, [4]*len(rat_list[1]), rat_list[3], c='k', alpha=0.6)
jitter(ax3, [5]*len(rat_list[2]), rat_list[4], c='k', alpha=0.6)

ax1.set_ylim([3.5,4])
ax2.set_ylim([-0.75,2.4])
ax3.set_ylim([-7,-6.5])

ax2.set_ylabel( "Ratio $\\frac{p2}{p1}$", fontsize=13)
ax3.set_xticklabels( labels, fontsize=13)
ax1.grid()
ax2.grid()
ax3.grid()
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax2.spines.bottom.set_visible(False)
ax3.spines.top.set_visible(False)

ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)
ax2.tick_params(top=False, bottom=False)

d = .5
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

ax2.plot([0, 1], [0, 0], transform=ax2.transAxes, **kwargs)
ax3.plot([0, 1], [1, 1], transform=ax3.transAxes, **kwargs)
pp.show()


# ISI test
isi = [300, 600, 900]
exp_isi = [300, 600, 900]
label_isi = [200, 500, 800] # NOTE: changed to conform to terminology used in rest of paper
exp_ratios = [0.8798, 0.15144, 1.38627]
exp_sem    = [0.11695, 0.03599, 0.77899]
exp_std    = list(map( lambda x: np.sqrt(5.0)*x, exp_sem))

exp_indiv_ratios = [ [0.58455, 1.06524, 0.93872, 0.63393, 1.17655],
                     [0.16907, 0.0794,  0.05658, 0.21459, 0.23757],
                     [0.46637, 0.75766, 0.31232, 0.92226, 4.47274]]


rat_list = [[],[],[],[],[]]
ax = pp.subplot(211)
ax.set_axisbelow(True)
pp.grid(zorder=0)
Imult = 1.0
Navg = 50
for k in range(len(isi)):
  temp = [np.zeros(Navg),np.zeros(Navg),np.zeros(Navg),np.zeros(Navg),np.zeros(Navg)]
  curve = np.zeros(N)
  for kk in range(Navg):
    val,X,Xif = aux_f( x0, ret_traj=True, isi_c=isi[k], Imult=Imult, noise_c=True)

    curve += Xif[1,:N]/float(Navg)
    pp.plot( fakeT[:N], Xif[1,:N], 'purple', alpha=0.025, zorder=1)

    for et in range(5):
      p1s = max( Xif[1, round(et*N):round(et*N + (600-1)/tstep)]) - 0.0*Xif[1,round(100/tstep)]
      p2s = max( Xif[1, round(et*N + (300+isi[k])/tstep):round((et+1)*N)]) - 0.0*Xif[1,round(100/tstep)]
      temp[et][kk] = (p2s/p1s)
  for et in range(5): rat_list[et].append( temp[et])
  pp.plot( fakeT[:N], curve, 'purple', zorder=2)

pp.plot( fakeT[:N], Imult*fakeOFC[:N], 'k',alpha=0.5, zorder=2)
pp.xlabel("Time (ms)")
pp.title( "Effect of changing ISI")

# PANEL Z: replace subplot with show, resize to about a third of a screen
#pp.subplot(212)
pp.show()

pp.errorbar( label_isi, list(map( np.mean, rat_list[0])), list(map( np.std, rat_list[0])), # was isi
                     capsize=7, color='b', ecolor='b', linestyle='--')
pp.plot( label_isi, list(map( np.mean, rat_list[0])), 'b*', linestyle='none') # was isi
pp.errorbar( label_isi, exp_ratios, exp_std, capsize=7, ecolor='k', color='k', linestyle='--') # was exp_isi
pp.plot( label_isi, exp_ratios, 'k*', linestyle='none') # was exp_isi
for k in range(5):
  pp.plot( label_isi, list(map( lambda x: x[k], exp_indiv_ratios)), 'ko', linestyle='none') # was exp_isi

pp.xlabel( "ISI (ms)", fontsize=13)
pp.ylabel( "Ratio $\\frac{p2}{p1} \pm \sigma$", fontsize=13)
pp.grid()
pp.legend(['Model inference', 'Experimental'], fontsize=13)
pp.show()
