DIR="$( cd "$( dirname "$BASH_SOURCE" )" && pwd -P )"

# build converter
CONVERTER=${DIR}/../build/Release/MNNConvert
if [ ! -e ${CONVERTER} ]; then
  echo "can't find ${CONVERTER}, building converter firstly "
  exit
fi

for f in $(find $1 -name '*.tflite' ); 
do 
$CONVERTER -f TFLITE --modelFile $f --MNNModel ${f%.*}.mnn --bizCode 0000
done