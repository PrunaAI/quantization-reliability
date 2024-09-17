dataset_expert_mapping = {
    'P101': 'Sociology, Stanford University',
    'P103': 'Linguistics, University of Cambridge',
    'P108': 'Business Administration, Wharton School',
    'P127': 'Business Law, NYU',
    'P1376': 'Political Geography, Sciences Po',
    'P1412': 'Historical Linguistics, University of Edinburgh',
    'P159': 'Business Geography, London School of Economics',
    'P17': 'Geography, Oxford University',
    'P176': 'Industrial Engineering, ETH Zurich',
    'P178': 'Innovation Management, INSEAD',
    'P19': 'History, Harvard University',
    'P20': 'History, Yale University',
    'P264': 'Music Industry, Berklee College of Music',
    'P27': 'Political Science, Columbia University',
    'P276': 'Geography, University of Chicago',
    'P30': 'Geography, University of California, Berkeley',
    'P364': 'Media Studies, USC',
    'P37': 'Linguistics, MIT',
    'P495': 'Cultural Studies, Sorbonne University',
    'P740': 'Organizational History, University of Tokyo'
}

dataset_example_questions = {
    'P17': [
        ("Where is France located?", "France is located in Western Europe."),
        ("What is the geographical location of Germany?", "Germany is located in Central Europe."),
        ("In which country can Tokyo be found?", "Tokyo can be found in Japan."),
        ("What is the capital city of Russia?", "The capital city of Russia is Moscow."),
        ("Where is Brazil geographically situated?", "Brazil is geographically situated in South America.")
    ],
    'P19': [
        ("Where was Albert Einstein born?", "Albert Einstein was born in Ulm, Germany."),
        ("What is William Shakespeare's birthplace?", "William Shakespeare's birthplace is Stratford-upon-Avon, England."),
        ("In what city was Nelson Mandela born?", "Nelson Mandela was born in Mvezo, South Africa."),
        ("What city was Marie Curie born in?", "Marie Curie was born in Warsaw, Poland."),
        ("In which location was Leonardo da Vinci born?", "Leonardo da Vinci was born in Vinci, Italy.")
    ],
    'P20': [
        ("In what place did Napoleon Bonaparte pass away?", "Napoleon Bonaparte passed away on the island of Saint Helena."),
        ("Where did Martin Luther King Jr. meet his end?", "Martin Luther King Jr. met his end in Memphis, Tennessee."),
        ("Which city did Elvis Presley depart this world?", "Elvis Presley departed this world in Memphis, Tennessee."),
        ("Where did Marie Antoinette breathe her last?", "Marie Antoinette breathed her last in Paris, France."),
        ("In which city did Leonardo da Vinci draw his last breath?", "Leonardo da Vinci drew his last breath in Amboise, France.")
    ],
    'P27': [
        ("What country is Lionel Messi a citizen of?", "Lionel Messi is a citizen of Argentina."),
        ("What country does Angela Merkel hold citizenship in?", "Angela Merkel holds citizenship in Germany."),
        ("What is Usain Bolt's nationality?", "Usain Bolt's nationality is Jamaican."),
        ("Which citizenship does Malala Yousafzai hold?", "Malala Yousafzai holds Pakistani citizenship."),
        ("What country is Elon Musk a national of?", "Elon Musk is a national of South Africa, Canada, and the United States.")
    ],
    'P30': [
        ("Which continent is Egypt located in?", "Egypt is located in Africa."),
        ("On which continent is India situated?", "India is situated in Asia."),
        ("In which continent can Brazil be found?", "Brazil can be found in South America."),
        ("To which continent does Australia belong?", "Australia belongs to the continent of Australia (also known as Oceania)."),
        ("What is the continental location of Germany?", "The continental location of Germany is Europe.")
    ],
    'P37': [
        ("What language is the official language of France?", "The official language of France is French."),
        ("What is the designated language of Japan?", "The designated language of Japan is Japanese."),
        ("What language holds official status in Brazil?", "Portuguese holds official status in Brazil."),
        ("What language is used as the primary means of communication in Russia?", "Russian is used as the primary means of communication in Russia."),
        ("What language is used for official documents in China?", "Standard Chinese (Mandarin) is used for official documents in China.")
    ],
    'P101': [
        ("What is Stephen Hawking's area of expertise?", "Stephen Hawking's area of expertise is theoretical physics and cosmology."),
        ("What is the primary focus of Jane Goodall's work?", "The primary focus of Jane Goodall's work is primatology and anthropology."),
        ("What is the field of specialization for Sigmund Freud?", "Sigmund Freud's field of specialization is psychoanalysis and psychology."),
        ("In what field is Marie Curie a specialist?", "Marie Curie is a specialist in the field of physics and chemistry."),
        ("What is Albert Einstein's principal area of study?", "Albert Einstein's principal area of study is theoretical physics.")
    ],
    'P103': [
        ("What is the native language of Pablo Picasso?", "The native language of Pablo Picasso is Spanish."),
        ("In what language did Leo Tolstoy grow up speaking?", "Leo Tolstoy grew up speaking Russian."),
        ("What is the first language of Mahatma Gandhi?", "The first language of Mahatma Gandhi is Gujarati."),
        ("Which language did Marie Curie learn as her mother tongue?", "Marie Curie learned Polish as her mother tongue."),
        ("What is Albert Einstein's mother tongue?", "Albert Einstein's mother tongue is German.")
    ],
    'P108': [
        ("Which organization does Satya Nadella work for?", "Satya Nadella works for Microsoft."),
        ("Who is Tim Cook's employer?", "Tim Cook's employer is Apple."),
        ("Which company employs Mary Barra?", "General Motors employs Mary Barra."),
        ("Where does Sundar Pichai work?", "Sundar Pichai works at Google."),
        ("What is the business that Mark Zuckerberg is working for?", "Mark Zuckerberg is working for Facebook (Meta).")
    ],
    'P127': [
        ("Which company is the owner of Instagram?", "Facebook (Meta) is the owner of Instagram."),
        ("Which company owns Pixar?", "Disney owns Pixar."),
        ("To which company does Jaguar belong?", "Jaguar belongs to Tata Motors."),
        ("Who is the controlling company of YouTube?", "Google (Alphabet) is the controlling company of YouTube."),
        ("What is the company that possesses LinkedIn?", "Microsoft is the company that possesses LinkedIn.")
    ],
    'P159': [
        ("In what city is Amazon headquartered?", "Amazon is headquartered in Seattle, Washington."),
        ("Which city is Apple's corporate headquarters located?", "Apple's corporate headquarters is located in Cupertino, California."),
        ("In which city is Toyota's head office?", "Toyota's head office is in Toyota City, Japan."),
        ("What city serves as the center of Facebook's operations?", "Menlo Park, California serves as the center of Facebook's operations."),
        ("In what city does Microsoft maintain its headquarters?", "Microsoft maintains its headquarters in Redmond, Washington.")
    ],
    'P176': [
        ("What is the manufacturer of the iPhone?", "Apple is the manufacturer of the iPhone."),
        ("Which company produces the Playstation?", "Sony produces the Playstation."),
        ("Which company manufactures the Model S?", "Tesla manufactures the Model S."),
        ("What is the company that makes Oreo cookies?", "Nabisco (owned by Mondelez International) makes Oreo cookies."),
        ("Who is responsible for the production of Lego bricks?", "The Lego Group is responsible for the production of Lego bricks.")
    ],
    'P178': [
        ("Which company is the creator of Windows?", "Microsoft is the creator of Windows."),
        ("Which organization is behind the development of Android?", "Google is behind the development of Android."),
        ("What is the company that developed the Java programming language?", "Sun Microsystems (now owned by Oracle) developed the Java programming language."),
        ("Which company is responsible for creating Photoshop?", "Adobe is responsible for creating Photoshop."),
        ("What is the name of the company that created the World Wide Web?", "CERN is the organization where Tim Berners-Lee created the World Wide Web.")
    ],
    'P264': [
        ("What is the record label for The Beatles?", "The Beatles were primarily signed to EMI's Parlophone label in the UK and Capitol Records in the US."),
        ("Which music label represents Taylor Swift?", "Taylor Swift is currently signed to Republic Records."),
        ("Under which music label is Beyoncé signed?", "Beyoncé is signed to Columbia Records."),
        ("Who is Eminem signed with in the music industry?", "Eminem is signed to Aftermath Entertainment and Shady Records, distributed by Interscope Records."),
        ("What is the name of the music label that manages BTS?", "BTS is managed by Big Hit Music, a subsidiary of HYBE Corporation.")
    ],
    'P276': [
        ("What is the location of the Louvre Museum?", "The Louvre Museum is located in Paris, France."),
        ("In what place is the Taj Mahal situated?", "The Taj Mahal is situated in Agra, India."),
        ("Where is the Great Wall of China located?", "The Great Wall of China is located in northern China, stretching across several provinces."),
        ("In which city is the Colosseum situated?", "The Colosseum is situated in Rome, Italy."),
        ("What is the geographic position of Machu Picchu?", "Machu Picchu is located in the Cusco Region of Peru, high in the Andes Mountains.")
    ],
    'P364': [
        ("What is the native language of 'Don Quixote'?", "The native language of 'Don Quixote' is Spanish."),
        ("What is the original tongue of 'The Iliad'?", "The original tongue of 'The Iliad' is Ancient Greek."),
        ("In which language was 'War and Peace' originally written?", "'War and Peace' was originally written in Russian."),
        ("What is the primary language of 'One Hundred Years of Solitude'?", "The primary language of 'One Hundred Years of Solitude' is Spanish."),
        ("Which language does 'The Tale of Genji' speak natively?", "'The Tale of Genji' speaks Japanese natively.")
    ],
    'P495': [
        ("Which country was pizza created in?", "Pizza was created in Italy."),
        ("In which nation was the automobile first established?", "The automobile was first established in Germany."),
        ("What is the country of origin of sushi?", "The country of origin of sushi is Japan."),
        ("Which nation is the birthplace of the Internet?", "The United States is the birthplace of the Internet."),
        ("In which country did blue jeans originate?", "Blue jeans originated in the United States.")
    ],
    'P740': [
        ("Which city was Google founded in?", "Google was founded in Menlo Park, California."),
        ("What is the founding city of Toyota?", "The founding city of Toyota is Toyota City (formerly Koromo), Japan."),
        ("In what locality did McDonald's originate?", "McDonald's originated in San Bernardino, California."),
        ("What was the city of origin for IKEA?", "The city of origin for IKEA is Älmhult, Sweden."),
        ("In which city was Nintendo established?", "Nintendo was established in Kyoto, Japan.")
    ],
    'P1376': [
        ("Which country's capital is Paris?", "Paris is the capital of France."),
        ("What administrative division has Sacramento as its capital?", "Sacramento is the capital of California, a state in the United States."),
        ("In which country is Tokyo the capital city?", "Tokyo is the capital city of Japan."),
        ("What is the country that has designated Canberra as its administrative capital?", "Australia has designated Canberra as its administrative capital."),
        ("In which administrative division is Albany the seat of government?", "Albany is the seat of government in New York, a state in the United States.")
    ],
    'P1412': [
        ("What language did Julius Caesar speak?", "Julius Caesar spoke Latin."),
        ("In what language did Confucius use to converse?", "Confucius used to converse in Classical Chinese."),
        ("Which language did Cleopatra previously employ for communication?", "Cleopatra previously employed Greek for communication, along with Egyptian."),
        ("What was the language that Dante Alighieri used to speak earlier?", "Dante Alighieri used to speak Italian (specifically, the Tuscan dialect) earlier."),
        ("Which language was once used by Johann Wolfgang von Goethe to communicate?", "Johann Wolfgang von Goethe used German to communicate.")
    ]
}