# wavmatch
This script is based on the wavmatchsynthesis script which slices two .wav files every specified amount of samples, retrieves MFCCs from each slice, compares the slices from the first input file with those from the second based on the MFCCs, and reconstructs the first input file using matched data from the second.

This manifestation of this idea differs in that it calculates the slice length by counting a specified amount of zero-crossings in the waveform instead of a given number of samples.  This results in unequal lengths which the script resolves by pasting the matched slices successively without matching to the exact timing of the input file.  This also results in silences in one channel toward the end of the output file due to uneven output lengths between stereo channels.

In order to run this script you will need a Nvidia GPU with cuda cores and you will need to be in a rapidsai conda environment.  Instructions for installing this environment are located at https://docs.rapids.ai/install

The `num_crossings` variable must be an even number in order to avoid DC offset.
