base_url=http://latent-composition.csail.mit.edu/other_projects/anyres_gan/pretrained
target_dir=pretrained2
mkdir -p $target_dir
wget $base_url/bird_pretrained_final.pkl -P $target_dir
wget $base_url/church_pretrained_final.pkl -P $target_dir
wget $base_url/ffhq6k_pretrained_final.pkl -P $target_dir
wget $base_url/mountain_pretrained_final.pkl -P $target_dir
wget $base_url/patch-bird-final-options.json -P $target_dir
wget $base_url/patch-church-final-options.json -P $target_dir
wget $base_url/patch-ffhq6k-final-options.json -P $target_dir
wget $base_url/patch-mountain-final-options.json -P $target_dir

wget $base_url/mountain_pano_pretrained_final.pkl -P $target_dir
