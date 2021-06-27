# Face_detection-AWS-Whatsapp-Mail

This program features some exclusive applications of face detection. It registers and recognizes faces of 2 persons and performs tasks. When it recognizes the face of person "1" as registered, it sends a mail to the mail ID taken as input, and a text to a person whose phone no. is added, both of which- text and phone no. is taken as input.

When it recognizes the face of "2" person, it launches an EC2 instance in AWS, creates a volume of 5GiB and attaches it to the instance.

NOTE: Make sure whatsapp web is logged in the browser, or you'll have to login first and due to which the wait time might exceed and the text is not delivered.
You'll have to provide credentials to your gmail account in the form of token, generated from your Google account under security- App password, and input them when prompted.
