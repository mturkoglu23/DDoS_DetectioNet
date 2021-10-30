# DDoS_DetectioNet

**A Novel Approach for Accurate Detection of the DDoS Attacks in SDN-based SCADA Systems Based on Deep Recurrent Neural Networks**

Supervisory Control and Data Acquisition (SCADA) systems supervise and monitor critical infrastructures and industrial processes. However, SCADA systems running on conventional network architecture have scalability and manageability limitations. Through its programmable dynamic architecture, Software-Defined Network (SDN) technology offers rapid configuration, scalability, and better manageability for SCADA systems. Combining existing SCADA systems with SDN has produced more practical SDN-based SCADA systems. However, due to their sensitive positions, SCADA systems are the targets of highly dangerous cyberattacks. In particular, failure to accurately detect and take action against cyberattacks like DDoS -Distributed Denial of Service- may lead to service disruption in SDN-based SCADA systems which may cause loss of life or massive financial losses. This study suggested the Recurrent Neural Network (RNN) classifier model, including two separate parallel deep learning-based methods, Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU), to better the detection of DDoS attacks targeting SDN-based SCADA systems. The proposed parallel structure was trained from end to end with a training dataset and tested with the validation dataset. This model was processed in the transfer learning procedure. The features were extracted with the training dataset, and the extracted features were classified with Support Vector Machines (SVM). While in transfer learning, the validation data was used in feature extraction and obtained features were classified with a trained SVM classifier. As part of the work, a sample dataset containing both DDoS attacks and regular network traffic data was created using an experimentally generated SDN-based SCADA topology. While experimental works yielded an accuracy of 97.62% for DDoS attack detection, transfer learning allowed a performance improvement of around 5%. The results have shown that the proposed RNN deep learning classifier model can effectively detect DDoS attacks targeting SDN-based SCADA systems. 
