Description
Goal of the Competition
The goal of this competition is to classify isolated American Sign Language (ASL) signs. You will create a TensorFlow Lite model trained on labeled landmark data extracted using the MediaPipe Holistic Solution.

Your work may improve the ability of PopSign* to help relatives of deaf children learn basic signs and communicate better with their loved ones.


Context
Every day, 33 babies are born with permanent hearing loss in the U.S.
Around 90% of which are born to hearing parents many of which may not know American Sign Language. (kdhe.ks.gov, deafchildren.org) Without sign language, deaf babies are at risk of Language Deprivation Syndrome. This syndrome is characterized by a lack of access to naturally occurring language acquisition during their critical language-learning years. It can cause serious impacts on different aspects of their lives, such as relationships, education, and employment.

Learning sign language is challenging.
Learning American Sign Language is as difficult for English speakers as learning Japanese. (jstor.org) It takes time and resources, which many parents don't have. They want to learn sign language, but it's hard when they are working long hours just to make ends meet. And even if they find the time and money for classes, the classes are often far away.

Games can help.
PopSign is a smartphone game app that makes learning American Sign Language fun, interactive, and accessible. Players match videos of ASL signs with bubbles containing written English words to pop them.
PopSign is designed to help parents with deaf children learn ASL, but it's open to anyone who wants to learn sign language vocabulary. By adding a sign language recognizer from this competition, PopSign players will be able to sign the type of bubble they want to shoot, providing the player with the opportunity to practice the sign themselves instead of just watching videos of other people signing.

You can help connect deaf children and their parents.
By training a sign language recognizer for PopSign, you can help make the game more interactive and improve the learning and confidence of players who want to learn sign language to communicate with their loved ones.

Why TensorFlow Lite
To allow the ML model to run on device in an attempt to limit latency inside the game, PopSign doesn’t send user videos to the cloud. Therefore, all inference must be done on the phone itself. PopSign is building its recognition pipeline on top of TensorFlow Lite, which runs on both Android and iOS. In order for the competition models to integrate seamlessly with PopSign, we are asking our competitors to submit their entries in the form of TensorFlow Lite models.

Special thanks to our partners
We’d like to thank the Georgia Institute of Technology, the National Technical Institute for the Deaf at Rochester Institute of Technology, and Deaf Professional Arts Network for their work to create the dataset, the PopSign game, and overall competition preparation.

This is a Code Competition. Refer to Code Requirements for details.

*PopSign is an app developed by the Georgia Institute of Technology and the National Technical Institute for the Deaf at Rochester Institute of Technology. The app is available in beta on Android and iOS.
^We cannot guarantee the competition will benefit the competitors or the disabled community directly.

Evaluation
The evaluation metric for this contest is simple classification accuracy.

Submission Process
In this competition you will be submitting a TensorFlow Lite model file. The model must take one or more landmark frames as an input and return a float vector (the predicted probabilities of each sign class) as the output. Your model must be packaged into a submission.zip file and compatible with the TensorFlow Lite Runtime v2.9.1. You are welcome to train your model using the framework of your choice, as long as you convert the model checkpoint into the tflite format prior to submission.

Your model must also perform inference with less than 100 milliseconds of latency per video on average and use less than 40 MB of storage space. Expect to see approximately 40,000 videos in the test set. We allow an additional 10 minute buffer for loading the data and miscellaneous overhead.

Each video is loaded with the following function:

ROWS_PER_FRAME = 543  # number of landmarks per frame

def load_relevant_data_subset(pq_path):
    data_columns = ['x', 'y', 'z']
    data = pd.read_parquet(pq_path, columns=data_columns)
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.values.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)
Inference is performed (roughly) as follows, ignoring details like how we manage multiple videos:

import tflite_runtime.interpreter as tflite
interpreter = tflite.Interpreter(model_path)

found_signatures = list(interpreter.get_signature_list().keys())

if REQUIRED_SIGNATURE not in found_signatures:
    raise KernelEvalException('Required input signature not found.')

prediction_fn = interpreter.get_signature_runner("serving_default")
output = prediction_fn(inputs=frames)
sign = np.argmax(output["outputs"])
Timeline
February 23, 2023 - Start Date.

April 24, 2023 - Entry Deadline. You must accept the competition rules before this date in order to compete.

April 24, 2023 - Team Merger Deadline. This is the last day participants may join or merge teams.

May 1, 2023 - Final Submission Deadline.

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. The competition organizers reserve the right to update the contest timeline if they deem it necessary.

Prizes
1st Place - $50,000
2nd Place - $20,000
3rd Place - $10,000
4th Place - $10,000
5th Place - $10,000
Code Requirements


This is a Code Competition
Submissions to this competition must be made through Notebooks. In order for the "Submit" button to be active after a commit, the following conditions must be met:

CPU Notebook <= 9 hours run-time
GPU Notebook <= 9 hours run-time
Internet access disabled
Freely & publicly available external data is allowed, including pre-trained models
Submission file must be named submission.zip.
Please see the Code Competition FAQ for more information on how to submit. And review the code debugging doc if you are encountering submission errors.

Acknowledgements
The dataset provided by Deaf Professional Arts Network and the Georgia Institute of Technology is licensed under CC-BY 4.0. Kaggle and Google do not own and have not validated the dataset in any way.

Data Card
Dataset Card for the Isolated Sign Language Recognition Corpus
Dataset Summary
The Isolated Sign Language Recognition corpus (version 1.0) is a collection of hand and facial landmarks generated by Mediapipe version 0.9.0.1 on ~100k videos of isolated signs performed by 21 Deaf signers from a 250-sign vocabulary.

Supported Tasks and Leaderboards
https://www.kaggle.com/competitions/asl-signs/leaderboard

Languages
American Sign Language

Dataset Structure
Data Instances
{'frame': 27, 'row_id': '27-face-0', 'type': 'face', 'landmark_index': 0, 'x': 0.4764270484447479, 'y': 0.3772650957107544, 'z': -0.05066078156232834}

Data Fields
See https://www.kaggle.com/competitions/asl-signs/data

Data Splits
Not applicable.

Dataset Creation
Curation Rationale
The signs in the dataset represent 250 of the first concepts taught to infants in any language. The goal is to create an isolated sign recognizer to incorporate into educational games for helping hearing parents of Deaf children learn American Sign Language (ASL). Around 90% of deaf infants are born to hearing parents, many of whom may not know American Sign Language. (kdhe.ks.gov, deafchildren.org). Surrounding Deaf children with sign helps avoid Language Deprivation Syndrome. This syndrome is characterized by a lack of access to naturally occurring language acquisition during the critical language-learning years. It can cause serious impacts on different aspects of their lives, such as relationships, education, and employment.
Learning American Sign Language (ASL) is as difficult for English speakers as learning Japanese (jstor.org). It takes time and resources that many parents don't have. They want to learn sign language, but it's hard when they are working long hours just to make ends meet. And even if they find the time and money for classes, the classes are often far away.
PopSign is a smartphone game app that makes learning American Sign Language fun, interactive, and accessible. Currently, players match videos of ASL signs with bubbles containing written English words to pop the bubbles and advance game play.
By adding isolated sign language recognition to Popsign, parents will play the game by making the signs instead of watching videos of signing. This sort of expressive practice improves confidence for communicating with Deaf children and the Deaf community.

Source Data
Initial Data Collection and Normalization
Signers who communicate using American Sign Language as their primary language were recruited from across the United States. They were shipped a Pixel 4a smartphone with an installed collection app. The app prompted the signer with the concept in English to sign, randomly selected from the 250-sign vocabulary. Signers pressed and held an on-screen button on the phone to record video while signing each concept, releasing the button after each sign. The video of the sign is extracted with a buffer 0.5 seconds before the press of the button and 0.5 seconds after the release of the button. This method of video collection matches the game interface where players touch the screen to aim a bubble and release the touch after they have finished signing.

While the app provided a video example of the sign desired, signers routinely made variants of the sign based on their background and region. More rarely, signers might fingerspell a sign, miss it completely, or produce the wrong sign. Extraneous movements, such as scratching an itch, or the ending movement from the previous sign or the onset of the next sign, are sometimes included. Conversely, some signers pressed the button late or released the button early, causing cropping in some sign examples. Some signers sign with their left hand; others sign with their right. Some signers switch their signing hand. All of these situations must be handled by the game’s recognition system.

While the game includes 250 signs, it only needs to distinguish between five signs at a time due to the game design. Since accuracy increases as vocabulary decreases, even a recognition system with 60% accuracy on the 250-sign task should perform well when distinguishing between five signs.

Who are the source language producers?
21 signers recruited by the Deaf Professional Arts Network provided the sign. They are from many regions across the United States and all use American Sign Language as their primary form of communication. They represent a mix of skin tones and genders.

Annotations
Annotation process
Each video was annotated at creation time by the smartphone app. Videos were coarsely reviewed to attempt to remove poor recordings, but little judgment was made on the correctness or quality of the sign itself.

Who are the annotators?
Researchers at the Georgia Institute of Technology coarsely reviewed the individual videos.

Personal and Sensitive Information
The landmark data has been de-identified. Landmark data should not be used to identify or re-identify an individual. Landmark data is not intended to enable any form of identity recognition or store any unique biometric identification.

Considerations for Using the Data
Social Impact of Dataset
The Isolated Sign Language Recognition corpus (version 1.0), which contains Mediapipe landmarks only, will be used to create sign language recognition systems for Popsign, an educational game that encourages hearing parents of deaf infants to practice their ASL signing. The same dataset can be used to add signing to other games. For example, one proposed use is to create a game that allows Deaf children to practice their written English skills. The video set upon which the corpus is based is being used to examine variations in signing and provide examples of those variations for the wider Deaf community.

Discussion of Biases
While ASL is the most common sign language used in the United States, there are many sign languages, including British Sign Language, Native American Sign Languages, Hawaiian Sign Language, French Sign Language, and Signed Exact English. In addition, there are many regional and cultural accents associated with sign in the United States, including Black Sign Language. This dataset focuses on American Sign Language, but it does not capture a representative sample of all the sign variations that would be commonly understood in conversation. ASL has a grammar that is very different from English, and isolated signs do not capture the variation that occurs when a concept is signed in context. A larger number of signers is necessary to better represent skin tones, hand features, and different levels of signing dexterity.

Other Known Limitations
This isolated sign dataset is intended to help create educational games for teaching ASL, and is not appropriate for other purposes such as ASL-to-English translation or natural language interfaces for computers.

Additional Information
Dataset Curators
The Deaf Professional Arts Network (DPAN), is a 501(c)(3) non-profit founded in 2006 to make music, entertainment, and media accessible. The Georgia Institute of Technology is a top-10 public research university committed to improving the human condition through advanced science and technology. The National Technical Institute for the Deaf is one of the nine colleges of the Rochester Institute of Technology and is home to the world’s first and largest technological college for deaf and hard-of-hearing students.

Licensing Information
The dataset provided by Deaf Professional Arts Network and the Georgia Institute of Technology is licensed under CC-BY.

Citation Information
See the bottom of the Kaggle competition overview page for citation information.

Contributions
Thanks to the staff at DPAN and the students and faculty at Georgia Tech and NTID who make Popsign and this dataset possible.

—-

Citation
Ashley Chow, Glenn Cameron, Mark Sherwood, Phil Culliton, Sam Sepah, Sohier Dane, and Thad Starner. Google - Isolated Sign Language Recognition. https://kaggle.com/competitions/asl-signs, 2023. Kaggle.

Dataset Description
Deaf children are often born to hearing parents who do not know sign language. Your challenge in this competition is to help identify signs made in processed videos, which will support the development of mobile apps to help teach parents sign language so they can communicate with their Deaf children.

This competition requires submissions to be made in the form of TensorFlow Lite models. You are welcome to train your model using the framework of your choice as long as you convert the model checkpoint into the tflite format prior to submission. Please see the evaluation page for details.

Update: the rerun dataset has been published here.

Files
train_landmark_files/[participant_id]/[sequence_id].parquet The landmark data. The landmarks were extracted from raw videos with the MediaPipe holistic model. Not all of the frames necessarily had visible hands or hands that could be detected by the model.

Landmark data should not be used to identify or re-identify an individual. Landmark data is not intended to enable any form of identity recognition or store any unique biometric identification.

frame - The frame number in the raw video.
row_id - A unique identifier for the row.
type - The type of landmark. One of ['face', 'left_hand', 'pose', 'right_hand'].
landmark_index - The landmark index number. Details of the hand landmark locations can be found here.
[x/y/z] - The normalized spatial coordinates of the landmark. These are the only columns that will be provided to your submitted model for inference. The MediaPipe model is not fully trained to predict depth so you may wish to ignore the z values.
train.csv

path - The path to the landmark file.
participant_id - A unique identifier for the data contributor.
sequence_id - A unique identifier for the landmark sequence.
sign - The label for the landmark sequence.