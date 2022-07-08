#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <time.h>
#include <sys/socket.h>
#include <math.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <csignal>


// DADA includes for this example
#include <futils.h>
#include <dada_def.h>
#include <dada_hdu.h>
#include <multilog.h>
#include <ipcio.h>
#include <ascii_header.h>

#include <stdio.h>
//#include <string.h>
#include <errno.h>
//#include <stdlib.h>
#include <unistd.h>
//#include <sys/types.h>
//#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
//#include "tsfifo.h"
//#include<linux/slab.h>
#include <fcntl.h>

#include <sys/time.h>
//#include <stdatomic.h>

#include <cuda_runtime.h>


typedef struct 
{
    /* DADA Logger */ 
    multilog_t * log;
    
    // default shared memory key
    key_t dada_key;

    // DADA Header + Data unit
    dada_hdu_t * hdu;

    uint64_t block_size;
    //uint64_t header_size = ipcbuf_get_bufsz (hdu->header_block);
    uint64_t *block_id;
    uint64_t *curbufsz;

}dada_CB;

void* init_dada(void)
{
    dada_CB* CB=(dada_CB*)malloc(sizeof(dada_CB));
    
    /* DADA Logger */ 
    CB->log = 0;
  
    // default shared memory key
    CB->dada_key = DADA_DEFAULT_BLOCK_KEY;
  
    CB->block_id=(uint64_t*)malloc(sizeof(uint64_t));
    CB->curbufsz=(uint64_t*)malloc(sizeof(uint64_t));
    
    // DADA Header + Data unit
    CB->hdu = 0;

    // create a multilogger
    CB->log = multilog_open ("example_dada_reader", 0);

    // set the destination for multilog to stdout
    multilog_add ( CB->log, stdout);

    // create the HDU struct
    CB->hdu = dada_hdu_create ( CB->log);

    // set the key to connecting to the HDU
    dada_hdu_set_key (CB->hdu, CB->dada_key);
    
    if (dada_hdu_connect (CB->hdu) != 0)
    {
        multilog (CB->log, LOG_ERR, "could not connect to HDU\n");
        return 0;
    }

    CB->block_size = ipcbuf_get_bufsz (&(CB->hdu->data_block->buf));
  
    //uint64_t header_size = ipcbuf_get_bufsz (hdu->header_block);
    *(CB->block_id) = 0;
    *(CB->curbufsz) = 0;
    multilog (CB->log, LOG_INFO, "block size=%d\n" ,CB->block_size);
  
    if (dada_hdu_lock_read(CB->hdu) != 0)
    {
    multilog (CB->log, LOG_ERR, "could not lock read on HDU\n");
    return 0;
    }
    
    return (void*)CB;
}

int exit_dada(void *CB_void)
{
    dada_CB *CB=(dada_CB*)CB_void;
    if (dada_hdu_unlock_read(CB->hdu) != 0)
    {
    multilog (CB->log, LOG_ERR, "could not unlock read on HDU\n");
    return EXIT_FAILURE;
    }
    if (dada_hdu_disconnect (CB->hdu) != 0)
    {
    multilog (CB->log, LOG_ERR, "could not disconnect from hdu\n");
    return EXIT_FAILURE;
    }
    
    multilog_close(CB->log);
    
    return EXIT_SUCCESS;
}

long long int read_dada (void *CB_void , char *input_gpu , long long int ask_size , char *stop_flag)
{
    dada_CB *CB=(dada_CB*)CB_void;
    long long int read_size=0;
    int loop_num = ask_size/CB->block_size;
    char* data_pointer=NULL;
    int block_num;
    int try_time=0;
    
    if (ask_size%(CB->block_size)!=0)
    {
        multilog (CB->log, LOG_ERR, "read_data fail , ask_size is not legal\n");
        exit(-1);
    }
    for(block_num=0;(block_num<loop_num)&&(*stop_flag!=1);block_num++)
    {
        while(try_time<5)
        {
            data_pointer=ipcio_open_block_read(CB->hdu->data_block, CB->curbufsz,CB->block_id);
            if (data_pointer==0)
            {
                sleep(0.1);
                try_time++;
            }
            else
            {
                try_time=0;
                break;
            }
        }
        
        if(data_pointer==0)
        {
            multilog (CB->log, LOG_ERR, "ipcio_open_block_read failed\n");
            return read_size;
        }
    
        cudaMemcpy(input_gpu,data_pointer,CB->block_size,cudaMemcpyHostToDevice);
    
        if (ipcio_close_block_read (CB->hdu->data_block, CB->block_size) < 0)
        {
            multilog (CB->log, LOG_ERR, "ipcio_close_block_read failed\n");
            return -1;
        }
        input_gpu=input_gpu+CB->block_size;
        read_size+=CB->block_size;
    }
    return read_size;
}


