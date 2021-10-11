/*#include "filterbank.h"*/
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* input and output files and logfile (filterbank.monitor) */
FILE *input, *output, *logfile;
char  inpfile[80], outfile[80];

/* global variables describing the data */
double time_offset;

/* global variables describing the operating mode */
float start_time, final_time, clip_threshold;

int obits, sumifs, headerless, headerfile, swapout, invert_band;
/*int compute_spectra, do_vanvleck, hanning, hamming, zerolagdump;*/
int zerolagdump=0;
/*int headeronly;*/
char ifstream[8];
char flip_band=0;
/*unsigned char *gmrtzap;*/

char rawdatafile[80], source_name[80];
int machine_id, telescope_id, data_type, nchans, nbits, nifs, scan_number,
  barycentric, pulsarcentric; /* these two added Aug 20, 2004 DRL */
double tstart,mjdobs,tsamp,fch1,foff,refdm,az_start,za_start,src_raj,src_dej;
/*double gal_l,gal_b,header_tobs,raw_fch1,raw_foff;*/
int nbeams, ibeam;
char isign;
/* added 20 December 2000    JMC */
/*double srcl,srcb;*/
/*double ast0, lst0;*/
/*long wapp_scan_number;*/
/*char project[8];*/
/*char culprits[24];*/
/*double analog_power[2];*/


/* added frequency table for use with non-contiguous data */
/* double frequency_table[4096]; /* note limited number of channels */
/* long int npuls; /* added for binary pulse profile format */

void filterbank_header(FILE *outptr) /* includefile */
{
  int i,j;
  output=outptr;
  if (obits == -1) obits=nbits;
  /* go no further here if not interested in header parameters */
  if (headerless) return;
  /* broadcast the header parameters to the output stream */
  if (machine_id != 0) {
    send_string("HEADER_START");
    send_string("rawdatafile");
    send_string(inpfile);
    if (!strings_equal(source_name,"")) {
      send_string("source_name");
      send_string(source_name);
    }
    send_int("machine_id",machine_id);
    send_int("telescope_id",telescope_id);
    send_coords(src_raj,src_dej,az_start,za_start);
    if (zerolagdump) {
      /* time series data DM=0.0 */
      send_int("data_type",2);
      refdm=0.0;
      send_double("refdm",refdm);
      send_int("nchans",1);
    } else {
      /* filterbank data */
      /* N.B. for dedisperse to work, foff<0 so flip if necessary */
      send_int("data_type",1);
      /*if (foff>0) {*/
      if (0) { /*don't invert the band -- changes made by Weiwei Zhu */
	flip_band=1;
	/* send a signal to the conversion signals to invert the band */
        send_double("fch1",fch1+foff*(nchans-1));
        send_double("foff",-1.0*foff);
      } else {
	/* no inversion necessary */
	flip_band=0;
        send_double("fch1",fch1);
        send_double("foff",foff);
      }
      send_int("nchans",nchans);
    }
    /* beam info */
    send_int("nbeams",nbeams);
    send_int("ibeam",ibeam);
    /*printf("in c code, ibeam:%d", ibeam);*/
    /* number of bits per sample */
    send_int("nbits",obits);
    /* start time and sample interval */
    send_double("tstart",tstart+(double)start_time/86400.0);
    send_double("tsamp",tsamp);
    if (sumifs) {
      send_int("nifs",1);
    } else {
      j=0;
      for (i=1;i<=nifs;i++) if (ifstream[i-1]=='Y') j++;
      if (j==0) error_message("no valid IF streams selected!");
      send_int("nifs",j);
    }
    send_string("HEADER_END");
  }
  
}

void write_header(char *outptr_in, char *source_name_in, int machine_id_in, int telescope_id_in, int nchans_in, int nbits_in, int nbeams_in, int ibeam_in, double tstart_in, double start_time_in,  double tsamp_in, double fch1_in, double foff_in, double az_start_in, double za_start_in, double src_raj_in, double src_dej_in){
/*strcpy(ifstream,"XXXX");*/
strcpy(inpfile, "");
obits = -1;
sumifs = 1;
headerless = 0;
FILE *outptr=fopen(outptr_in,"wb" );
strcpy(source_name, source_name_in);
machine_id = machine_id_in;
telescope_id = telescope_id_in;
nchans = nchans_in;
nbits = nbits_in;
nbeams = nbeams_in;
ibeam = ibeam_in;
tstart = tstart_in;
start_time = start_time_in;
tsamp = tsamp_in;
fch1 = fch1_in;
foff = foff_in;
az_start = az_start_in;
za_start = za_start_in;
src_raj = src_raj_in;
src_dej = src_dej_in;
filterbank_header(outptr);
fclose(outptr);
}

