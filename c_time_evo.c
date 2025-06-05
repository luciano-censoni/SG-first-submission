#include <math.h>
#include <stdio.h>
#include <stdlib.h>


#define BETA 10.0
#define TH 0.0
#define DELAY_AAC 12.0
#define SOUND_DUR 20.0
#define DELTA_TRANS 1030
#define MAX(x, y) (x > y) ? x : y


#ifdef FOR_WINDOWS
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif


double H( double t);
double f( double x, double beta, double th);
double I( long j, double t, double ptime);
double IOGsu( long j, double t, double B, int etype);
double IOGsd( long j, double t, double B, int etype);
double IOGpu( long j, double t, double B, int etype);
double IOGpd( long j, double t, double B, int etype);


DLLEXPORT double corr( double * x, double * y, long N);
DLLEXPORT double dist( double * x, double * y, long N);
DLLEXPORT double time_evo( double alpha_a, double alpha_e, double alpha_s, double alpha_p, double alpha_v,
  double Waa, double Wea, double Wpa,
  double Wee, double Wse, double Wpe, double Wve,
  double Wes, double Wps, double Wvs,
  double Wep, double Wpp, double Wvp,
  double Wsv,
  double WsOGu, double WsOGd,
  double WpOGu, double WpOGd,
  double BsOG, double BpOG,
  double Aext, double Eext, double Sext, double Pext, double Vext,
  long N, double t0, double tstep, double ptime, double Imult, int lfp_flag, int obj_type, int noise_flag,
  double * tgt_AAC, double * tgt_OFC0, double * tgt_OFC1, double * tgt_OFC2, double * tgt_OFC3, double * tgt_OFC4,
  double * retA, double * retE, double * retS, double * retP, double * retV,
  double * retAif, double * retEif, double * retSif, double * retPif, double * retVif );


double H( double t){ return (t >= 0.0) ? 1.0 : 0.0;}

double f( double x, double beta, double th){ return 1.0 / (1.0 + exp(-beta * (x-th))) - 1.0/(1.0 + exp(beta*th));}

double I( long j, double t, double ptime)
{
  if( j != 0) return 0.0;
  else
  {
    return 0.1*( H(t-(200.0+DELAY_AAC))-H(t-(200.0+SOUND_DUR+DELAY_AAC)) + H(t-(200.0+ptime+DELAY_AAC))-H(t-(200.0+ptime+SOUND_DUR+DELAY_AAC)));
  }
}

double IOGsu( long j, double t, double B, int etype)
{
  if( j != 2 || etype != 1) return 0.0;
  else return 0.1*H(t-200.0)*(B+(10.0-10.0*B)/(t-190.0));
}

double IOGsd( long j, double t, double B, int etype)
{
  if( j != 2 || etype != 2) return 0.0;
  else return -0.1* H(t-200.0)*(B+(10.0-10.0*B)/(t-190.0));
}

double IOGpu( long j, double t, double B, int etype)
{
  if( j != 3 || etype != 3) return 0.0;
  else return 0.1*H(t-200.0)*(B+(10.0-10.0*B)/(t-190.0));
}

double IOGpd( long j, double t, double B, int etype)
{
  if( j != 3 || etype != 4) return 0.0;
  else return -0.1* H(t-200.0)*(B+(10.0-10.0*B)/(t-190.0));
}

DLLEXPORT double corr( double * x, double * y, long N)
{
  double xm = 0.0;
  double ym = 0.0;
  for (long i = 0; i < N; i++)
  {
    xm += x[i];
    ym += y[i];
  }

  xm = xm / N;
  ym = ym / N;

  double sx  = 0.0;
  double sy  = 0.0;
  double sp  = 0.0;
  for (long i = 0; i < N; i++)
  {
    sx += pow(x[i] - xm, 2);
    sy += pow(y[i] - ym, 2);
    sp += (x[i] - xm) * (y[i] - ym);
  }

  return -1*(sp / sqrt( sx * sy ));
}


DLLEXPORT double dist( double * x, double * y, long N)
{
  double dd  = 0.0;
  for (long i = 0; i < N; i++) dd += pow( x[i] - y[i], 2);
  return dd/10.0;
}


DLLEXPORT double time_evo( double alpha_a, double alpha_e, double alpha_s, double alpha_p, double alpha_v,
  double Waa, double Wea, double Wpa,
  double Wee, double Wse, double Wpe, double Wve,
  double Wes, double Wps, double Wvs,
  double Wep, double Wpp, double Wvp,
  double Wsv,
  double WsOGu, double WsOGd,
  double WpOGu, double WpOGd,
  double BsOG, double BpOG,
  double Aext, double Eext, double Sext, double Pext, double Vext,
  long N, double t0, double tstep, double ptime, double Imult, int lfp_flag, int obj_type, int noise_flag,
  double * tgt_AAC, double * tgt_OFC0, double * tgt_OFC1, double * tgt_OFC2, double * tgt_OFC3, double * tgt_OFC4,
  double * retA, double * retE, double * retS, double * retP, double * retV,
  double * retAif, double * retEif, double * retSif, double * retPif, double * retVif )
{
  double Wsa = 0.0;
  double Wva = 0.0;
  double Wae = 0.0;
  double Was = 0.0;
  double Wss = 0.0;
  double Wap = 0.0;
  double Wsp = 0.0;
  double Wav = 0.0;
  double Wev = 0.0;
  double Wpv = 0.0;
  double Wvv = 0.0; // all these by definition

  double M[5][5] = { { Waa, Wae, Was, Wap, Wav},
                     { Wea, Wee, Wes, Wep, Wev},
                     { Wsa, Wse, Wss, Wsp, Wsv},
                     { Wpa, Wpe, Wps, Wpp, Wpv},
                     { Wva, Wve, Wvs, Wvp, Wvv} };

  double A[5] = {alpha_a, alpha_e, alpha_s, alpha_p, alpha_v};

  double * Xp0  = (double *) malloc( N * sizeof(double) );
  double * Xp1  = (double *) malloc( N * sizeof(double) );
  double * Xp2  = (double *) malloc( N * sizeof(double) );
  double * Xp3  = (double *) malloc( N * sizeof(double) );
  double * Xp4  = (double *) malloc( N * sizeof(double) );
  double * X0   = (double *) malloc( N * sizeof(double) );
  double * X1   = (double *) malloc( N * sizeof(double) );
  double * X2   = (double *) malloc( N * sizeof(double) );
  double * X3   = (double *) malloc( N * sizeof(double) );
  double * X4   = (double *) malloc( N * sizeof(double) );
  double * X0if = (double *) malloc( N * sizeof(double) );
  double * X1if = (double *) malloc( N * sizeof(double) );
  double * X2if = (double *) malloc( N * sizeof(double) );
  double * X3if = (double *) malloc( N * sizeof(double) );
  double * X4if = (double *) malloc( N * sizeof(double) ); 

  if( Xp0  == NULL) exit(1);
  if( Xp1  == NULL) exit(1);
  if( Xp2  == NULL) exit(1);
  if( Xp3  == NULL) exit(1);
  if( Xp4  == NULL) exit(1);
  if( X0   == NULL) exit(1);
  if( X1   == NULL) exit(1);
  if( X2   == NULL) exit(1);
  if( X3   == NULL) exit(1);
  if( X4   == NULL) exit(1);
  if( X0if == NULL) exit(1);
  if( X1if == NULL) exit(1);
  if( X2if == NULL) exit(1);
  if( X3if == NULL) exit(1);
  if( X4if == NULL) exit(1);

  double * Xp[5]  = {Xp0, Xp1, Xp2, Xp3, Xp4};
  double * X[5]   = {X0, X1, X2, X3, X4};
  double * ret_vec[5] = {retA, retE, retS, retP, retV};
  double * ret_vec_if[5] = {retAif, retEif, retSif, retPif, retVif};
  double * Xif[5] = {X0if, X1if, X2if, X3if, X4if};

  double * tgt_OFC[5] = { tgt_OFC0, tgt_OFC1, tgt_OFC2, tgt_OFC3, tgt_OFC4};

  double val = 0.0;

  double tt;
  double Iext[5] = {0.0, 0.0, 0.0, 0.0, 0.0};

  for( int exp_type = 0; exp_type < 5; exp_type++)
  {
    for( int j = 0; j < 5; j++)
    {
      Xp[j][0]  = 0.0;
      Xif[j][0] = 0.0;
      ret_vec_if[j][N*exp_type] = 0.0;
      X[j][0] = 0.0;
      ret_vec[j][N*exp_type] = 0.0;
    }

    tt = t0;
    for( long k = 1; k < N; k++)
    {
      tt += tstep;

      for( int j = 0; j < 5; j++)
      {
        X[j][k]  = X[j][k-1]  + tstep * Xp[j][k-1];
        ret_vec[j][ N*exp_type +k] = X[j][k]; // length assumed to be 5N
      }

      for( int j = 0; j < 5; j++)
      {
        Xp[j][k] = 0.0;
        for( int m = 0; m < 5; m++)
          Xp[j][k] += M[j][m] * X[m][k-1];
      }

      Xp[1][k] -= M[1][0]*X[0][k-1];
      Xp[1][k] += M[1][0]*X[0][MAX(k-1-DELTA_TRANS, 0)];
      Xp[3][k] -= M[3][0]*X[0][k-1];
      Xp[3][k] += M[3][0]*X[0][MAX(k-1-DELTA_TRANS, 0)];

      for( int j = 0; j < 5; j++)
      {
        Xif[j][k] = Xp[j][k] +WsOGu*IOGsu(j,tt-tstep,BsOG,exp_type) +WsOGd*IOGsd(j,tt-tstep,BpOG,exp_type) +WpOGu*IOGpu(j,tt-tstep,BsOG,exp_type) +WpOGd*IOGpd(j,tt-tstep,BpOG,exp_type) +Iext[j] +Imult*I(j,tt-tstep,ptime);
        ret_vec_if[j][ N*exp_type +k] = Xp[j][k]; // length assumed to be 5N
      }

      for( int j = 0; j < 5; j++)
        Xp[j][k] = A[j]*( f( Xp[j][k] +Imult*I(j,tt-tstep,ptime) +WsOGu*IOGsu(j,tt-tstep,BsOG,exp_type) +WsOGd*IOGsd(j,tt-tstep,BpOG,exp_type) +WpOGu*IOGpu(j,tt-tstep,BsOG,exp_type) +WpOGd*IOGpd(j,tt-tstep,BpOG,exp_type) +Iext[j],BETA, TH) -X[j][k-1] );

      if( noise_flag) X[1][k] += ((double) rand()/(1.0+RAND_MAX) -0.5)*0.5*1e-3;

    } // for k

    // calculate val
    switch( obj_type)
    {
      case 0: // corr
        if( exp_type == 0) val += corr( Xif[0], tgt_AAC, N); // only add this for baseline experiment
        val += corr( Xif[1], tgt_OFC[exp_type], N);
      break;
      case 1: // dist
        if( exp_type == 0) val += dist( Xif[0], tgt_AAC, N); // only add this for baseline experiment
        val += dist( Xif[1], tgt_OFC[exp_type], N);
      break;
    }
  } // for exp_type

  free( Xp0);
  free( Xp1);
  free( Xp2);
  free( Xp3);
  free( Xp4);
  free( X0);
  free( X1);
  free( X2);
  free( X3);
  free( X4);
  free( X0if);
  free( X1if);
  free( X2if);
  free( X3if);
  free( X4if);
  return val;
}

