username=$1
password=$2
echo -e "\nDownloading FLAME..."
mkdir -p DECA/data/FLAME2020/
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&sfile=FLAME2020.zip&resume=1' -O './FLAME2020.zip' --no-check-certificate --continue
unzip FLAME2020.zip -d DECA/data/
rm -f FLAME2020.zip
