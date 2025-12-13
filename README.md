find /home/dataset-assist-1/SevereWeather_AI_2025/CP/TestSetB -maxdepth 1 -mindepth 1 -type d | xargs -I {} -P 32   rsync -aW --ignore-existing {} ./TestSetB

nohup bash run.scwds.mamba.sh train > train_mamba_scwds.log 2>&1 &