## WGAN Overview
The paper begins with the question ‘what does it mean to learn a probability distribution in unsupervised learning?’. VAE and GAN takes the approach of estimating Pr(real data distribution) by passing a random variable Z with a fixed distribution p(z) that follows the distribution **Pθ  to generate the distribution close to Pr**. 

Under this context, the paper suggests a new way of defining a **distance(divergence) between ρ(Pθ, Pr)** . The way of computing the distance between the distributions is critical because it is highly connected to **‘the convergence of distributions’**. Weak distance makes continuous mapping from θ → Pθ easier which leads to a distribution convergence. This is why the matter of keeping θ → Pθ continuous can be interpreted as **making  θ → ρ(Pθ, Pr) continuous**.

> The convergence of the distributions Pθt depends on the way we **compute the distance between distributions**…
> 

---

### Earth-Mover(EM) distance

> “γ(x, y) indicates how much “mass”must be transported from x to y in order to transform the distributions from Pr to the distribution Pg. The **EM distance then is the “cost” of the optimal transport plan.”**
> 

![Untitled](https://user-images.githubusercontent.com/86555104/217251226-f05693dd-54f9-4422-9d6e-919bbbff1c03.png)

The author logically proves how EM distance satisfies the ‘continuity’ compared to other divergence methods such as JS, KL and TV divergences.

![Untitled (1)](https://user-images.githubusercontent.com/86555104/217251243-5efe2175-3d63-492a-aed5-1cef841b3cbe.png)

The left **(EM) plot results a continuous loss function and provides usable gradient anywhere,** while JS plot shows a disjoint section at the middle. However,  W(Pr, Pg) needs to satisfy the Lipschitz constraint to apply its ‘continuity’ feature to the neural network. (Assumptions below)  

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c60ff630-00ac-464b-9477-95285263d635/Untitled.png)

### Wasserstein GAN

Authors previously proved how useful EM distance is. So, the next step is to show **‘how to optimize the EM distance**’ . This question can be answered by rewriting W(**Pθ, Pr)** with Kantorovich-Rubinstein duality. (Refer to the paper’s appendix to find out more about the mathematical process)

![Untitled (2)](https://user-images.githubusercontent.com/86555104/217251286-524cfd1a-1169-4a5e-a7c2-14d19935e30a.png)

f(x) is so called ‘critic’ which is a key concept of WGAN. WGAN  tries to get more reliable gradient by training critic until optimality makes the mode collapse impossible.

![Untitled (2)](https://user-images.githubusercontent.com/86555104/217251310-740bfe7a-01e8-4d9f-98be-742cd6b2bf97.png)

(weight clipping is used to enhance Lipschitz constraint)

### Benefits of WGAN

- Drastically reduces mode collapse phenomenon and likey to make more robust model than GAN by enhancing the stability
