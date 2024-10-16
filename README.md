# Reduced_cs_measurements_for_reconstruction
Trained neural networks reduce total measurements needed to reconstruct optical signals from 100% to 8%


Moreover, it can further be shown that the number of compressively sensed measurements needed to accurately classify the image can be significantly reduced using machine learning. By only using 30 measurements (instead of 400), the 20x20 image could be classified with 100% accuracy. This was one of the main results of this research and was also a big motivation to try machine learning over other algorithms for compressive sensing. Lr changed from 0.001 to 0.0011

This has profound implications as oftentimes we want to quickly know the object we are imaging and relying on all CS measurements could take time to acquire. This can significantly boost the image classification time as we don't need to wait for all 400 measurements to be completed. Furthermore, the agent built could be modified to provide a live prediction of the class as measurements come in.
