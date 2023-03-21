# Covid-detection-FL
Final Year Project 


<h3>Commands</h3> 
<br>python client1.py 9969</br>
<br>python client2.py 9969</br>
<br>python server.py 9969</br>

<h3>FED_AVG ALgorithms</h3>
In Federated Learning (FL), aggregation algorithms are used to combine the model updates from different devices or parties to create a global model. The choice of aggregation algorithm depends on various factors such as the nature of the data, the privacy requirements, and the performance goals of the FL system.

For chest X-rays, there are several aggregation algorithms that could be used, such as:

Federated Averaging (FedAvg): This is the most commonly used aggregation algorithm in FL. It involves computing the average of the model updates from each device or party and using this average to update the global model.

Federated Stochastic Gradient Descent (FedSGD): This algorithm is similar to FedAvg but uses stochastic gradient descent (SGD) to optimize the global model. It is more computationally efficient than FedAvg but may require more communication between devices.

Secure Aggregation: This algorithm is designed to ensure the privacy of the model updates by using cryptographic techniques to aggregate the updates without revealing their contents.

Federated Exponentially Weighted Aggregation (FedEWA): This algorithm uses an exponential weighting scheme to give more weight to recent model updates. It is designed to improve the convergence speed of the FL system.

The choice of aggregation algorithm for chest X-rays will depend on factors such as the number of devices or parties involved, the computational resources available, and the privacy requirements of the FL system. It is important to carefully consider these factors when selecting an aggregation algorithm to ensure that the FL system performs optimally while maintaining the privacy and security of the data.
