# filmfinder

This project aims to build a film recommandation problem based on your Letterbox Profile.

## Input:

We use two dabatases: https://www.kaggle.com/datasets/samlearner/letterboxd-movie-ratings-data?select=users_export.csv with 8139 users, 11M scores

https://www.kaggle.com/datasets/gsimonx37/letterboxd?select=movies.csv with 795 000 films

After data cleaning and merging, we arrive at 7,2k users, 25k films with descriptions and 5M triplets (user,film,score). 
The sparse matrix is then 97% empty.

## Model

We use Factorization Machines.
$$ y(x) = w_0 + \sum^n_i{w_i x_i} + \sum^n_i\sum^n_{j=i+1}{<v_i,v_j>x_i,x_j} $$

where w0 is the baseline score
wi is the weight of caracteristic i
vi is the latent vector of caracteristic i, of size k (hyperparameters)

Let observation be: Alice saw Titanic and gave it a 4.
x is a 1-hot encoding of the observation(Alice,Titaic) and y = 4.

$$ 4 = w_0 + w_{alice} + w_{titanic} + <v_{alice},v_{Titanic}>$$

Loss is :
$$ \[
\mathcal L=
\frac12\sum_{(x,y)\in D}
\bigl(\hat y(x)-y\bigr)^2
+\frac{\lambda_w}{2}\|w\|^2
+\frac{\lambda_V}{2}\|V\|_F^2
\]$$


```