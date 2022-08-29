# Wasserstein-GAN
Pytorch Implementation of Wasserstein GAN

Original Paper : https://arxiv.org/pdf/1701.07875.pdf



EM distance (Wasserstein distance)


![image](https://user-images.githubusercontent.com/69974410/187249143-b3aaba14-c066-48ca-ad8f-502a23881227.png)



Continuity and Differentiability of Wasserstein distance


![image](https://user-images.githubusercontent.com/69974410/187249248-e60ed047-f880-40f4-8b0c-6b2f28d1c18c.png)




Because of Kantorovich-Rubinstein duality, Wasserstein distance can be transformed into


![image](https://user-images.githubusercontent.com/69974410/187249561-9ad3a745-4e1c-475a-9612-e3d13a9bc528.png)

(satisfies  1-Lipschitz continuity)


Final loss

![image](https://user-images.githubusercontent.com/69974410/187250240-ac965484-0634-4f22-b79f-76aaf6a5e513.png)
