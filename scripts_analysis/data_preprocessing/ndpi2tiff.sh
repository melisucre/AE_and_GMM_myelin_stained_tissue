
#!/bin/bash

set -e

microscope='ISM'
animal='38'
stain='mye'
what='mtbi'

# directory containing ndpi files
input_dir=".../data/${microscope}_${animal}/${stain}"
# file_prefix="${microscope}-${animal}-15 ${stain}"
file_prefix="${microscope}-${animal}"


# output directory
output_dir=".../AE/image_preprocessing/data/${what}/${microscope}_${animal}/tiff_files/${stain}"
    
# create that directory if it doesn't exist
mkdir -p "${output_dir}"

# variable to keep track of the glass_number
glass_number=1

# loop through ndpi files in the input directory
for ndpi_file in "${input_dir}"/"${file_prefix}"*.ndpi; do

    output_path="${output_dir}/glass_${glass_number}"

    # Check if the output directory already exists
    if [ ! -d "$output_path" ]; then
        mkdir -p "${output_path}"
        
        # Convert the NDPI file to the desired format
        ndpisplit -O "${output_path}" -x0.625 "${ndpi_file}"
        ndpisplit -O "${output_path}" -x40 "${ndpi_file}"
    fi

    glass_number=$((glass_number + 1))

done

