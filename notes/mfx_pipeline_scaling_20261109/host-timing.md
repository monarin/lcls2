| Section (exclusive host wall time) | Seconds | Main-thread CPU seconds |
|---|---:|---:|
| kvikio_wait | 4.5235 | 1.4210 |
| cpu_dgram_construction | 3.4835 | 3.4653 |
| input_group_issue | 2.6591 | 2.6298 |
| cpu_ancillary_read | 2.5367 | 2.5106 |
| descriptor_build_upload_call.enqueue | 2.1881 | 2.1747 |
| parse_groups_call.enqueue | 2.0116 | 1.9869 |
| kvikio_submit | 1.7851 | 1.6660 |
| gpu_batch_controller | 1.6243 | 1.6074 |
| slot_submit_including_null_wait | 1.1545 | 1.1471 |
| cpu_eventmanager_initialization | 1.1331 | 1.1265 |
| subbatch_planning | 1.0417 | 1.0351 |
| consumer_retirement_release | 0.5871 | 0.5841 |
| step_handling | 0.5498 | 0.5468 |
| input_group_parse_submit | 0.5241 | 0.5196 |
| expand_calibrate_call | 0.3365 | 0.3251 |
| mpi_batch_receive_unpack_request | 0.1989 | 0.1741 |
| run_event_materialization | 0.1930 | 0.1921 |
| gpu_pipeline_initialization | 0.1820 | 0.1747 |
| cube_bins_upload_accumulate_call | 0.1563 | 0.1557 |
| events_iterator | 0.1133 | 0.1119 |
| file_epoch_resolution | 0.0532 | 0.0524 |
| attach_gpu_event | 0.0359 | 0.0360 |
| expansion_kernel.enqueue | 0.0265 | 0.0262 |
| cube_count_kernel.enqueue | 0.0250 | 0.0246 |
| calibration_kernel.enqueue | 0.0226 | 0.0222 |
| cube_sum_kernel.enqueue | 0.0165 | 0.0163 |
| producer_retirement_wait | 0.0130 | 0.0130 |
| slot_flush_active | 0.0058 | 0.0057 |
| gpu_pipeline_close | 0.0029 | 0.0029 |
| step_handling.close | 0.0025 | 0.0024 |
| gpu_batch_controller.close | 0.0025 | 0.0024 |
| slot_flush_active.close | 0.0000 | 0.0000 |
| mpi_batch_receive_unpack_request.close | 0.0000 | 0.0000 |
| Outside instrumented sections | 0.2400 | not measured |
