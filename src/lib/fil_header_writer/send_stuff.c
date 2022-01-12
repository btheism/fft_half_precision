/*#ifdef HAVE_CONFIG_H*/
/*#include <config.h>*/
/*#endif*/
#include <stdio.h>
#include <string.h>
/*#include "sigproc.h"*/
extern FILE *input, *output;
extern int swapout;

//先写入了string的长度，再写入string本身
void send_string(char *string) /* includefile */
{
  int len;
  len=strlen(string);
  if (swapout) swap_int(&len);
  //把string的占用空间（即变量len）写入文件头，且变量len已根据swapout的值事先进行了大小端转换，写了一个int
  fwrite(&len, sizeof(int), 1, output);
  //再把len转回去！简直多次一举，下一行直接写成fwrite(string, sizeof(char), strlen(string), output)不就行了
  if (swapout) swap_int(&len);
  //把string写入文件头
  fwrite(string, sizeof(char), len, output);
  /*fprintf(stderr,"%s\n",string);*/
}

void send_float(char *name,float floating_point) /* includefile */
{
  send_string(name);
  if (swapout) swap_float(&floating_point);
  fwrite(&floating_point,sizeof(float),1,output);
  /*fprintf(stderr,"%f\n",floating_point);*/
}

//写入值的名称后写入值的数值
void send_double (char *name, double double_precision) /* includefile */
{
  send_string(name);
  if (swapout) swap_double(&double_precision);
  fwrite(&double_precision,sizeof(double),1,output);
  /*fprintf(stderr,"%f\n",double_precision);*/
}

void send_int(char *name, int integer) /* includefile */
{
  send_string(name);
  if (swapout) swap_int(&integer);
  fwrite(&integer,sizeof(int),1,output);
  /*fprintf(stderr,"%d\n",integer);*/
}

void send_char(char *name, char integer) /* includefile */
{
  send_string(name);
  fwrite(&integer,sizeof(char),1,output);
}


void send_long(char *name, long integer) /* includefile */
{
  send_string(name);
  if (swapout) swap_long(&integer);
  fwrite(&integer,sizeof(long),1,output);
  /*fprintf(stderr,"%ld\n",integer);*/
}

void send_coords(double raj, double dej, double az, double za) /*includefile*/
{
  if ((raj != 0.0) || (raj != -1.0)) send_double("src_raj",raj);
  if ((dej != 0.0) || (dej != -1.0)) send_double("src_dej",dej);
  if ((az != 0.0)  || (az != -1.0))  send_double("az_start",az);
  if ((za != 0.0)  || (za != -1.0))  send_double("za_start",za);
}
