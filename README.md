# Experiments on Open Principal Odor Map - With intensity mappings

Please find the original readme from the main openpom repo.

The purpose of this Readme is to track my progress with different experiments to understand the inner-workings of the POM model hopefully towards narrowing down on my Main Goals. 
I'm also using this project as a way to improve and sharpen my skills in applied Statistics, so any feedback would be greatly appreciated!

## Main goals:
Creating a systematic way to create an ‘engine’ capable of digitally conceptualizing odours (with the eventual goal of synthesis)
 -- Find a reliable minimum value for ‘n’ such that ‘n’ separate odour dimensions can represent the entire odour space.
 -- Find a chemical compound/mixture for each of these n dimensions.

## QSOR relationship between compounds and odours:
We rely on the POM GNN model that seems to have shown promising abilities in mapping a QSOR between chemical structure of compound to some corresponding perception labels.
Also, the dataset used to train this model appears to cover a large spectrum of odour labels in the human scent experience. Hence, we are assuming that this model has created a comprehensive understanding of odours. I’m working on work done from the openpom port/replication of the POM paper.
   ### Testing out the setup and visualizing model learnings
   1. Visualizations of embedding space of different predetermined ‘primary odour’ categories to verify the learnings of the openpom port of the POM research paper. Here’s the Crocker Henderson categorization: <img width="586" alt="image" src="https://github.com/Maadi5/openpom/assets/55384421/d46bda2b-ddf2-4e17-866f-b5a49981f619">
   2. Visualizing the heatmaps of the intensity of certain odours from the Intensity dataset mapped against the 2-dim PCA coordinates of the POM model embedding: <img width="824" alt="image" src="https://github.com/Maadi5/openpom/assets/55384421/3a07bab5-1b8a-4ff6-90ef-4c2b6145ccdb">
   3. Attempts to model this regression dataset (from the data above) using different techniques ranging from gradient boosting to neural networks: <img width="400" alt="image" src="https://github.com/Maadi5/openpom/assets/55384421/6316e11c-7813-45f3-b5d3-acdc29087cf3">
   4. To do: Look at Activation maps instead of embeddings. Try to look at patterns (like number of neurons activated, for different intensities of certain odours.) This is a work in progress..


      
   



## Contributors of the original openpom repo:
**Aryan Amit Barsainyan**, National Institute of Technology Karnataka, India: code, data cleaning, model development<br/>
**Ritesh Kumar**, CSIR-CSIO, Chandigarh, India: data cleaning, hyperparameter optimisation<br/>
**Pinaki Saha**, University of Hertfordshire, UK: discussions and feedback<br/>
**Michael Schmuker**, University of Hertfordshire, UK: conceptualisation, project lead<br/>

## References:
\[1\] A Principal Odor Map Unifies Diverse Tasks in Human Olfactory Perception.<br/>

Brian K. Lee, Emily J. Mayhew, Benjamin Sanchez-Lengeling, Jennifer N. Wei, Wesley W. Qian, Kelsie A. Little, Matthew Andres, Britney B. Nguyen, Theresa Moloy, Jacob Yasonik, Jane K. Parker, Richard C. Gerkin, Joel D. Mainland, Alexander B. Wiltschko<br/>

Science381,999-1006(2023).DOI: [10.1126/science.ade4401](https://doi.org/10.1126/science.ade4401) <br/>
bioRxiv 2022.09.01.504602; doi: [https://doi.org/10.1101/2022.09.01.504602](https://doi.org/10.1101/2022.09.01.504602)
