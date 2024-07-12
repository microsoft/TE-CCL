#! /bin/bash

# Run teccl solve for all configurations in the experiments directory

for topology in NDv2 DGX2;
do
    for chassis in experiments/"$topology"*/;
    do
        for collective in "$chassis"*/;
        do
            for epoch_type in "$collective"*/;
            do
                for config_file in $epoch_type*/*;
                do
                    echo Running teccl solve --input_args $config_file
                    teccl solve --input_args $config_file
                done
            done
        done
    done
done

