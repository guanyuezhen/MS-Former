
python train.py --file_root BGMix --inWidth 256 --inHeight 256 --patch_size 256 --memory_length 128 --depth 3 --lr 5e-4 --batch_size 32 --max_steps 20000

python train.py --file_root SYSU --inWidth 256 --inHeight 256 --patch_size 128 --memory_length 128 --depth 3 --lr 5e-4 --batch_size 32 --max_steps 20000

python train.py --file_root LEVIR --inWidth 256 --inHeight 256 --patch_size 128 --memory_length 128 --depth 3 --lr 5e-4 --batch_size 32 --max_steps 20000



python train.py --file_root BGMix --inWidth 256 --inHeight 256 --patch_size 128 --memory_length 128 --depth 3 --lr 5e-4 --batch_size 32 --max_steps 20000

python train.py --file_root BGMix --inWidth 256 --inHeight 256 --patch_size 64 --memory_length 128 --depth 3 --lr 5e-4 --batch_size 32 --max_steps 20000

python train.py --file_root BGMix --inWidth 256 --inHeight 256 --patch_size 32 --memory_length 128 --depth 3 --lr 5e-4 --batch_size 32 --max_steps 20000

python train.py --file_root SYSU --inWidth 256 --inHeight 256 --patch_size 64 --memory_length 128 --depth 3 --lr 5e-4 --batch_size 32 --max_steps 20000

python train.py --file_root LEVIR --inWidth 256 --inHeight 256 --patch_size 64 --memory_length 128 --depth 3 --lr 5e-4 --batch_size 32 --max_steps 20000

python train.py --file_root SYSU --inWidth 256 --inHeight 256 --patch_size 32 --memory_length 128 --depth 3 --lr 5e-4 --batch_size 32 --max_steps 20000

python train.py --file_root LEVIR --inWidth 256 --inHeight 256 --patch_size 32 --memory_length 128 --depth 3 --lr 5e-4 --batch_size 32 --max_steps 20000


python test.py --file_root LEVIR --inWidth 1024 --inHeight 1024 --patch_size 128 --memory_length 128 --depth 3 --lr 5e-4 --batch_size 1 --max_steps 20000

python test.py --file_root LEVIR --inWidth 1024 --inHeight 1024 --patch_size 64 --memory_length 128 --depth 3 --lr 5e-4 --batch_size 1 --max_steps 20000

python test.py --file_root LEVIR --inWidth 1024 --inHeight 1024 --patch_size 32 --memory_length 128 --depth 3 --lr 5e-4 --batch_size 1 --max_steps 20000


