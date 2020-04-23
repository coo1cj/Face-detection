# Face-detection
Recently, i have a project about face detection. So i want to share it in git. 
After i make it better i'll push it in my git. This time, i won't use the any pre-trained classifier online. I'll use the hog + linearsvm to detecte the face. Maybe it's not a very good choice, so if i have more time, i'll use more methods to see which is better. 

So in this project, i've changed the parameters C, dual, weight_class of LinearSVC.
And from now on, i've found that when C = 1, dual = False, weight_class = None it's the best choice, but because of the different numbers between positif samples and negatif samples, i thought weight_class = 'balanced' was better, the fact is not that. 

I will retry it.
