#!/bin/bash
cd "$(dirname "$0")"
set -e

cd ../

budget_model_size=1e6
budget_flops=160e6
max_layers=18
population_size=512
evolution_max_iter=480000  # we suggest evolution_max_iter=480000 for

save_dir=../../save_dir/TE_VoV_cifar_params1M_flops160M
mkdir -p ${save_dir}

echo "SuperConvK3BNRELU(3,8,1,1)SuperVoVK3L3(8,16,1,8,1)SuperVoVK3L3(16,32,2,16,1)SuperVoVK3L3(32,64,2,32,1)SuperVoVK3L3(64,64,2,32,1)SuperConvK1BNRELU(64,128,1,1)" \
> ${save_dir}/init_plainnet.txt

python evolution_search.py --gpu 0 \
  --zero_shot_score NASWOT \
  --fix_initialize \
  --origin \
  --search_space SearchSpace/search_space_VoV.py \
  --budget_model_size ${budget_model_size} \
  --max_layers ${max_layers} \
  --batch_size 64 \
  --input_image_size 32 \
  --plainnet_struct_txt ${save_dir}/init_plainnet.txt \
  --num_classes 10 \
  --evolution_max_iter ${evolution_max_iter} \
  --population_size ${population_size} \
  --save_dir ${save_dir}


python analyze_model.py \
  --input_image_size 32 \
  --num_classes 10 \
  --arch Masternet.py:MasterNet \
  --plainnet_struct_txt ${save_dir}/best_structure.txt \
  > ${save_dir}/analyze_model.txt
#  --budget_flops ${budget_flops} \