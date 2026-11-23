#!/bin/bash

#input='/nas/data/processed/people/sos/composites_output/racmo/whole_domain_GrIS_ci/'
#'rh2m','tskin','v10m','psurf','u10m','q2m','qii','qli','qvi','t2m','lwsd','lwsu','swsu','swsd','ci','clcov','alb','precip','latf','senf','ff10m'
# for var in {'rh2m','tskin','v10m','psurf','u10m','q2m','qii','qli','qvi','t2m','lwsd','lwsu','swsu','swsd','clcov','alb','precip','latf','senf','ff10m'}; do
#     # cd $input/$var/highs
#     # rm *.nc
#     # cd $input/$var/lows
#     # rm *.nc
#     # cd $input/$var/pvalues
#     # rm *.nc
#     # cd $input/$var/seasons
#     # rm *.nc
#     cd $input/$var/pvalues
#     rm *5th_95th_perc.nc
#     continue
# done

# --------------------------
# input='/nas/data/processed/people/sos/composites_output/era5'
# # Declare an array of string with type
# declare -a StringArray=('usurf' 'vsurf' 'u500hPa' 'v500hPa')

# for var in ${StringArray[@]}; do
#     cd $input/$var/highs
#     rm *.nc
#     cd $input/$var/lows
#     rm *.nc
#     cd $input/$var/pvalues
#     rm *.nc
#     cd $input/$var/seasons
#     rm *.nc
#     continue
# done

# --------------------------
input='/nas/data/processed/people/sos/racmo_GrIS_domain/cross_sections/rearrange_cube/surface_level/'
# Declare an array of string with type
declare -a StringArray=('qii' 'qli' 'v10m' 'u10m')

for var in ${StringArray[@]}; do
    cd $input/$var/
    rm *.nc
    continue
done
# --------------------------

# ./racmo_GrIS_domain/preprocessing_codes

# /composites_output/era5/usurf
# /composites_output/era5/vsurf
# /composites_output/era5/u500hPa
# /composites_output/era5/v500hPa
# /composites_output/era5/usurf
# ./racmo_GrIS_domain/cross_sections/rearrange_cube/surface_level/qii
# ./racmo_GrIS_domain/cross_sections/rearrange_cube/surface_level/qli
# /racmo_GrIS_domain/cross_sections/rearrange_cube/surface_level/v10m
# /racmo_GrIS_domain/cross_sections/rearrange_cube/surface_level/u10m
# /racmo_GrIS_domain/surface_levels/v10m
# ./racmo_GrIS_domain/surface_levels/u10m

# for var in {'alb','latf','senf','lwsu','lwsd','qvi','swsu','swsd'}; do
#     cd $input/$var
#     rm -r *
#     echo "deleted ...."$input/$var
#     continue
# done
   
