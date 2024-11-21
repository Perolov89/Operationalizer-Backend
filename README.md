# Operationalizer Backend
A project to process images and figure out what’s on them using a trained CNN model. Specifically, it recognizes numbers (0-9) and math operators (+, -, *, /).

### Why I Made This
I wanted to learn about tensorflow and build something straightforward that could process/interpret images. Hence the notebook has step by step instructions and explanations for learning purposes.


### Deployment Challenges
To make it work on Vercel, I had to trim down the dependencies as much as possible and set up a new repo with a onnx model for the deployment: [hosting_operationalizer_backend](https://github.com/Perolov89/hosting_operationalizer_backend/tree/main). It’s stripped down to the basics but still works.





FrontEnd repo: [operationalizer](https://github.com/Perolov89/operationalizer)
