so_names=(
    backends/paddle/libpaddle_inference.so
)
for so in ${so_names[@]}; do
    patchelf --set-rpath '$ORIGIN' $so;
done