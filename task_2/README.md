# Federated Learning!

Hi! We’ll be learning about a machine learning technique called *federated learning* through the course of your second task as part of the Systems for ML assignment. Don’t worry if you’ve never heard of it, the core idea is pretty simple, and I’m sure you’ll pick it up quickly!

*A quick note: I'll point you to a lot of resources as we go ahead, and it's totally not necessary to go through everything completely! Just try to understand all the concepts involved, and finally apply them in the task given. You can follow any of the listed resources, as well as anything else you find online (which might be better!). Just have fun learning about the exciting domain of FL!*

## The roots: Distributed ML

Distributed machine learning is an umbrella term for any method that uses multiple computers to train a single machine learning model. This allows for the processing of larger datasets and more complex models. It achieves this by partitioning the data or the model itself across a network of machines, which then work in parallel to speed up the training process. This approach is crucial for both building highly complex AI models that would exceed the capacity of a single computer, as well as for running ML algorithms on a set of edge devices (computing devices located near the source of data, such as sensors, smartphones, cameras, router, etc.) which do not have much computational power in a single node.

Please go through this article for details if you're interested - [Distributed ML by IBM](https://www.ibm.com/think/topics/distributed-machine-learning)  
You can see federated learning listed under "data parallelism"!

## The idea: Federated learning

In federated learning, a model is trained across multiple ‘decentralized’ devices, like smartphones or servers, without exchanging local data. How? Instead of moving data to a central location, the model is sent to the data! The models are trained locally on each device, and then only the model updates are sent back to a central server for aggregation. This approach keeps user data private, enhancing security and privacy, while still enabling the development of a shared, improved AI model.

Google introduced this idea in their paper from 2016 titled "Communication-Efficient Learning of Deep Networks from Decentralized Data". So, rightfully, here's a video from Google Cloud explaining the main idea behind FL! - [Federated Learning by Google Cloud Tech](https://www.youtube.com/watch?v=X8YYWunttOY)

Here's the paper, it might be worth a read - [FL Origin Paper on arXiv](https://arxiv.org/abs/1602.05629)  
If you scroll through the paper, you'll see a section titled "The FederatedAveraging Algorithm" (or FedAvg in short) - *SPOILER: this is your task!*

There are many real-world use-cases for FL, you can take a look at some of them for inspiration! (Credits to *Flower.ai*)
- [FL in Healthcare](https://flower.ai/industries/healthcare/)
- [FL in Automotive](https://flower.ai/industries/automotive/)
- [FL in IoT Systems](https://flower.ai/industries/iot/)
- [FL in Finance](https://flower.ai/industries/finance/)

## The strategy: FedAvg

FedAvg is the most basic (and foundational, as you just saw) algorithm for federated learning. A server in this context refers to the central aggregator in the FL system and clients refer to the edge devices running the models, like your mobile phones!  
Let me list down the major steps involved in the algorithm:
- *Server initializes and sends the model.* The central server begins with a global model and sends a copy of its current parameters (weights and biases) to a selection of participating client devices.
- *Clients train models locally.* Each selected client trains the model on its own private dataset, performing multiple steps of a local optimization algorithm, such as stochastic gradient descent. Because each device's dataset is unique, each client will produce slightly different model updates.
- *Clients send updates back.* After completing their local training, the clients send their updated model parameters back to the central server. Critically, only the model updates are sent, and the raw training data never leaves the device.
- *Server aggregates the updates.* The server collects the model updates from all participating clients and computes a weighted average of their parameters. The weighting is typically proportional to the number of data points each client trained on, ensuring that a client with more data has a proportionally larger influence on the new global model.
- *Process repeats.* The server uses the aggregated average as the new global model. This new model is then sent back to clients for the next training round, and the cycle continues until the model converges.

## The implementation: PyTorch

PyTorch is an open-source machine learning library built on Python. It is one of the most popular frameworks in use at present due to its flexibility, ease-of-use, wide compatibility and vibrant community. It's great if you're already a part of this community, but it's not too late to join if you aren't!
- You can refer to the official PyTorch documentation, which is pretty well-written, whenever you face difficulties - [PyTorch Docs](https://docs.pytorch.org/docs/stable/index.html)
- There's also an official "Deep Learning with PyTorch" quickstart, which is quite nice - [PyTorch: A 60 Minute Blitz](https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- Finally, there's an extremely useful and beginner-friendly resource to learn PyTorch by Daniel Bourke - [learnpytorch.io](https://www.learnpytorch.io/)  
It'll take quite some time to go through the entire course, but you can save the link (you'll thank me later!).
- Of course, with the rise of LLMs, they're our best friends and teachers! But, the old-school docs are still a great resource if you dive deeper.

## Your task!

Oof, hope that wasn't too much to take at once (maybe it was, sorry!). Finally, coming to your task! *(drumroll)*  
It's pretty simple: implement FedAvg using basic PyTorch!

Here's something my friend Atharva found that should help you quite a bit with the task! - [FedAvg by HowDev](https://how.dev/answers/what-is-federated-averaging-fedavg)  
It's a brief walkthrough of the task, along with a toy implementation using NumPy (though it's almost exactly what you need to do). You just need to scale it up to full-sized models, i.e., use PyTorch instead of NumPy!

The important part here is to understand the fundamentals of federated learning (and hence, FedAvg) and the way models are accessed/manipulated/stored in PyTorch. Once you've done that, believe me, it won't take you a long time to do it!

*Another quick note: The task is intentionally open-ended. You're free to set up any common ML workload to demonstrate and test your FL implementation. If you're out of ideas or need a reference, feel free to check out the CIFAR10 implementation from the Extras section. For the server, you can have anything from a basic dictionary-based aggregator to a complete network-based architecture! Be creative and experiment with different ways you could do it (provided you have enough time). The more 'realistic' your setup can be, the better!*

## Submission

Just create a private repo according to the instructions on the [main branch](https://github.com/DaSH-Lab-CSIS/DaSH-Lab-Induction-Assignment-2025/tree/main) and code up your implementation in [this folder](https://github.com/DaSH-Lab-CSIS/DaSH-Lab-Induction-Assignment-2025/tree/ML-sys/task_2)!

Setting up PyTorch for the first time might be painful if you've never done it before. You can follow the instructions on this page to set it up with CUDA support - [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)

## Revision / Extras

If you need a quick revision of these concepts, the team at *Flower.ai* has a nice article summarising these ideas - [Federated Learning by Flower](https://flower.ai/docs/framework/tutorial-series-what-is-federated-learning.html)

*SPOILER: Flower is something we use all the time to set up our federated learning workloads!*

So, if you're curious about how we set up and run such FL workloads at scale, you can check out this video explaining the implementation of a model for the *CIFAR10* dataset on Flower - [FL Quickstart with Flower & PyTorch](https://www.youtube.com/watch?v=jOmmuzMIQ4c)

## Signing off

Hope you had fun solving the assignment! You probably learnt a lot of new skills as you went through it (I hope so! Reading research papers, writing maintainable code, etc. are all difficult skills to master!). Even if you don't get through the induction process, please don't be disappointed. Hope you continue exploring computer science research with this great start!

*A final note: If we see consistent effort and a good understanding, you have a good shot at qualifying even if your implementation is incomplete. So, make sure to submit whatever you've tried! We'd love to see your progress and discuss it in further rounds. For any doubts regarding the assignment, feel free to contact me (Pranav M R) or anyone else at the lab! Looking forward to seeing you at DaSH!*
