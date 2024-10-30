pinecone_sample_data = [
    {"id": "anime1", "text": "Naruto is an action-packed anime that follows Naruto Uzumaki, a young ninja with dreams of becoming the strongest Hokage of his village. Throughout his journey, Naruto faces various challenges, builds friendships, and strives for acceptance in a world that often shuns him. The series delves into themes of perseverance, loyalty, and the importance of bonds between people. With its rich character development and intense battles, Naruto has become a beloved staple in the anime community.", "genre": "Action, Adventure"},
    
    {"id": "anime2", "text": "One Piece tells the adventurous tale of Monkey D. Luffy, a spirited young pirate whose dream is to find the ultimate treasure known as the One Piece and become the King of the Pirates. Joined by a diverse crew, Luffy travels across the Grand Line, facing powerful enemies, uncovering hidden treasures, and experiencing the true meaning of friendship and freedom. Known for its imaginative world-building and humor, One Piece has captivated audiences worldwide and continues to be a cornerstone of anime culture.", "genre": "Adventure, Fantasy"},
    
    {"id": "anime3", "text": "Attack on Titan presents a gripping narrative set in a world where humanity faces extinction at the hands of gigantic humanoid creatures called Titans. The story follows Eren Yeager, who vows to eliminate the Titans after witnessing the destruction of his hometown. As he joins the military alongside his friends Mikasa and Armin, the series explores themes of survival, sacrifice, and the struggle for freedom. With its intense action sequences and emotional depth, Attack on Titan has garnered critical acclaim and a massive fanbase.", "genre": "Action, Drama, Fantasy"},
    
    {"id": "anime4", "text": "My Hero Academia revolves around a world where almost everyone possesses superpowers known as 'quirks.' The story follows Izuku Midoriya, a quirkless boy who dreams of becoming a hero. After meeting his idol, All Might, he inherits the legendary hero's power and enrolls in U.A. High School to train as a hero. The series showcases Midoriya's growth as he faces various challenges, builds friendships, and learns the true meaning of heroism, emphasizing the importance of determination and self-belief.", "genre": "Action, Superhero, School"},
    
    {"id": "anime5", "text": "Death Note follows the story of Light Yagami, a brilliant high school student who discovers a mysterious notebook that allows him to kill anyone whose name he writes in it. Seeking to rid the world of criminals, Light adopts the persona of 'Kira' and quickly gains notoriety. However, his actions attract the attention of a brilliant detective known only as 'L.' This psychological thriller explores complex themes of morality, justice, and the consequences of wielding absolute power, making it a must-watch for fans of suspenseful storytelling.", "genre": "Mystery, Thriller, Supernatural"},
    
    {"id": "anime6", "text": "Spirited Away, a masterpiece by Studio Ghibli, tells the enchanting story of a young girl named Chihiro who becomes trapped in a mystical spirit world. As she embarks on a quest to save her parents, who have been transformed into pigs, Chihiro must navigate through a realm filled with spirits, witches, and magical creatures. The film beautifully explores themes of identity, courage, and the importance of environmental consciousness, showcasing stunning animation and heartfelt storytelling that resonates with audiences of all ages.", "genre": "Fantasy, Adventure"},
    
    {"id": "anime7", "text": "Sword Art Online explores the dark consequences of virtual reality gaming when thousands of players become trapped in a popular MMORPG. Kirito, a skilled gamer, must navigate the challenges of surviving in this new world while uncovering the mysteries behind the game. As he forms alliances and battles formidable foes, the series raises questions about the nature of reality, relationships, and the impact of technology on human connections. Sword Art Online is known for its action-packed sequences and emotional depth.", "genre": "Action, Adventure, Fantasy"},
    
    {"id": "anime8", "text": "Demon Slayer follows the journey of Tanjiro Kamado, a kind-hearted boy whose family is slaughtered by demons. To save his sister Nezuko, who has been turned into a demon, Tanjiro becomes a demon slayer. The series showcases breathtaking animation and intense battle scenes, highlighting the importance of family bonds, determination, and compassion in the face of adversity. With its emotional storytelling and unique characters, Demon Slayer has quickly risen to popularity, captivating audiences around the world.", "genre": "Action, Fantasy, Supernatural"},
    
    {"id": "anime9", "text": "Fullmetal Alchemist: Brotherhood follows brothers Edward and Alphonse Elric on their quest to restore their bodies after a failed alchemical experiment. As they search for the Philosopher's Stone, the series explores themes of sacrifice, redemption, and the consequences of ambition. Set in a richly crafted world where alchemy reigns, the anime balances intense action with deep emotional moments and philosophical questions about the nature of humanity and the value of life, making it a classic among anime fans.", "genre": "Adventure, Fantasy, Action"},
    
    {"id": "anime10", "text": "One Punch Man is a satirical take on the superhero genre, focusing on Saitama, an overpowered hero who can defeat any opponent with a single punch. Despite his extraordinary abilities, Saitama struggles with boredom and a lack of recognition. The series cleverly parodies traditional superhero tropes while delivering thrilling action sequences and humor. With its unique premise and memorable characters, One Punch Man has garnered a dedicated fanbase and is celebrated for its witty storytelling and animation quality.", "genre": "Action, Comedy, Parody"},
    
    {"id": "anime11", "text": "Your Name is a beautifully crafted romantic fantasy film that tells the story of two teenagers, Taki and Mitsuha, who mysteriously swap bodies. As they navigate each other's lives, they form a deep connection despite never having met. The film masterfully intertwines themes of love, fate, and the passage of time, enhanced by stunning animation and a captivating soundtrack. Your Name has resonated with audiences worldwide, making it one of the highest-grossing anime films in history and a must-see for romance fans.", "genre": "Romance, Fantasy, Drama"},
    
    {"id": "anime12", "text": "Tokyo Ghoul follows Ken Kaneki, a college student who becomes a half-ghoul after a chance encounter with one. As he grapples with his new identity, Kaneki navigates a dark world filled with danger and moral ambiguity. The series explores themes of humanity, identity, and the struggle for acceptance, featuring intense action and psychological depth. With its unique blend of horror and drama, Tokyo Ghoul has captivated audiences and sparked discussions about the nature of good and evil.", "genre": "Horror, Action, Supernatural"},
    
    {"id": "anime13", "text": "Neon Genesis Evangelion is a groundbreaking mecha anime that delves into the psychological struggles of its characters as they pilot giant robots to combat mysterious beings known as Angels. The series explores existential themes, mental health, and the complexities of human relationships. With its innovative storytelling, philosophical undertones, and iconic visuals, Evangelion has left a lasting impact on the anime industry and continues to be a subject of analysis and discussion among fans and critics alike.", "genre": "Mecha, Psychological, Drama"},
    
    {"id": "anime14", "text": "Mob Psycho 100 follows Shigeo 'Mob' Kageyama, a powerful esper who strives for a normal life while dealing with his immense psychic abilities. The series blends humor, action, and supernatural elements as Mob navigates the challenges of adolescence and self-acceptance. With its unique animation style and heartfelt storytelling, Mob Psycho 100 emphasizes personal growth and the importance of emotions, making it a beloved series among fans of all ages.", "genre": "Action, Comedy, Supernatural"},
    
    {"id": "anime15", "text": "Fate/stay night revolves around Shirou Emiya, a young mage who finds himself embroiled in the Holy Grail War, a battle between legendary heroes. As he fights alongside his servant, Saber, he must confront moral dilemmas and the complexities of desire and heroism. The series combines action, romance, and fantasy elements, captivating audiences with its rich lore and character-driven storytelling. Fate/stay night has spawned numerous adaptations and remains a significant entry in the fantasy anime genre.", "genre": "Action, Fantasy, Romance"},
    
    {"id": "anime16", "text": "Haikyuu!! centers on Shoyo Hinata, a high school student who aspires to become a great volleyball player despite his short stature. Joined by a talented team, Hinata learns the values of teamwork, determination, and perseverance. The series highlights intense matches and character development, emphasizing the importance of sportsmanship and friendship. Haikyuu!! has gained a large following for its engaging storytelling, dynamic animation, and relatable characters, making it one of the top sports anime in recent years.", "genre": "Sports, Comedy, Drama"},
    
    {"id": "anime17", "text": "Monster follows Dr. Kenzo Tenma, a brilliant brain surgeon whose life takes a dark turn when he saves the life of a young boy who grows up to be a sociopathic killer. As Tenma embarks on a quest to stop the monster he inadvertently created, the series explores profound themes of morality, guilt, and the nature of evil. With its intricate plot and psychological depth, Monster is regarded as one of the best thriller anime, captivating viewers with its suspenseful storytelling.", "genre": "Psychological, Thriller, Drama"},
    
    {"id": "anime18", "text": "Steins;Gate revolves around a group of friends who discover a way to send messages into the past using a modified microwave. Their experiments lead to unintended consequences, drawing them into a complex web of time travel and alternate realities. The series expertly balances humor, drama, and suspense while exploring themes of friendship, sacrifice, and the ethical implications of time travel. Steins;Gate is celebrated for its intricate plot and character development, making it a standout in the sci-fi genre.", "genre": "Sci-Fi, Thriller, Drama"},
    
    {"id": "anime19", "text": "The Promised Neverland follows a group of children living in an idyllic orphanage who discover a dark secret about their fate. As they learn that they are raised as livestock for demons, they devise a plan to escape and survive. The series masterfully blends horror, mystery, and psychological elements, exploring themes of innocence, survival, and the loss of childhood. With its thrilling plot twists and emotional depth, The Promised Neverland has captured the hearts of many anime fans.", "genre": "Horror, Mystery, Thriller"},
    
    {"id": "anime20", "text": "Vinland Saga is an epic tale inspired by historical events, following Thorfinn, a young Viking warrior on a quest for revenge against Askeladd, his father's killer. Set in the brutal world of Norse sagas, the series explores themes of violence, honor, and redemption as Thorfinn navigates his desire for vengeance and the pursuit of a true warrior's path. With its stunning animation and character-driven storytelling, Vinland Saga has garnered critical acclaim for its depth and historical accuracy.", "genre": "Action, Adventure, Historical"},
    
    {"id": "anime21", "text": "Clannad is a heartwarming slice-of-life anime that follows Tomoya Okazaki, a delinquent who meets various girls at his school, each with their own struggles. As Tomoya forms relationships with these girls, particularly Nagisa, he learns about friendship, love, and the importance of family. The series masterfully combines humor and drama, culminating in an emotional narrative that explores themes of growth, loss, and the bonds that tie people together. Clannad is beloved for its relatable characters and touching moments.", "genre": "Drama, Romance, Slice of Life"},
    
    {"id": "anime22", "text": "Monster Musume follows the humorous misadventures of Kuruso Kimihito, a young man who becomes the reluctant caretaker of various monster girls who have entered into a cultural exchange program. The series blends harem comedy with fantasy elements, as Kuruso navigates the challenges of living with these unique creatures while dealing with the comedic and often chaotic situations that arise. Monster Musume offers a lighthearted take on relationships and acceptance, making it a fun watch for fans of the genre.", "genre": "Comedy, Fantasy, Harem"},
    
    {"id": "anime23", "text": "Sword Art Online: Alicization explores Kirito's journey as he finds himself trapped in a new virtual world known as Underworld. As he navigates this intricate world, he meets new friends and faces powerful foes, all while uncovering a dark conspiracy. The series delves into themes of friendship, identity, and the nature of consciousness. With its action-packed scenes and emotional storytelling, Sword Art Online: Alicization expands on the beloved franchise while exploring deeper philosophical questions.", "genre": "Action, Adventure, Fantasy"},
    
    {"id": "anime24", "text": "Your Lie in April follows Kōsei Arima, a piano prodigy who loses his ability to hear the sound of his piano after the death of his mother. His life changes when he meets Kaori Miyazono, a spirited violinist who helps him rediscover his passion for music. The series beautifully intertwines themes of love, loss, and the healing power of music. With its stunning animation and emotional depth, Your Lie in April is a touching exploration of grief and the impact of relationships.", "genre": "Drama, Romance, Music"},
    
    {"id": "anime25", "text": "Fairy Tail follows the adventures of Natsu Dragneel, a young wizard in search of the dragon Igneel. He joins the Fairy Tail guild, where he forms bonds with other wizards and embarks on numerous quests. The series emphasizes the importance of friendship, loyalty, and the bonds formed through shared experiences. With its vibrant characters and thrilling battles, Fairy Tail captivates audiences with its blend of action, humor, and heartfelt moments, making it a fan favorite in the fantasy genre.", "genre": "Action, Adventure, Fantasy"},
    
    {"id": "anime26", "text": "Jujutsu Kaisen follows Yuji Itadori, a high school student who becomes involved in the world of curses after swallowing a cursed object. He joins the Tokyo Metropolitan Jujutsu Technical School, where he learns to harness his newfound powers and fight curses alongside skilled sorcerers. The series blends action, horror, and supernatural elements while exploring themes of life, death, and the value of human connections. With its stunning animation and dynamic fight scenes, Jujutsu Kaisen has quickly gained popularity among anime fans.", "genre": "Action, Supernatural, Horror"},
    
    {"id": "anime27", "text": "Hellsing Ultimate follows the Hellsing Organization, led by Sir Integra Hellsing, as they battle supernatural threats, particularly vampires. The series centers around Alucard, a powerful vampire who works for Hellsing, and his conflicts with other supernatural beings. Known for its dark themes, intense action, and intricate storytelling, Hellsing Ultimate explores the nature of humanity and monstrosity. Its unique animation style and compelling characters make it a standout in the horror and action genres.", "genre": "Action, Horror, Supernatural"},
    
    {"id": "anime28", "text": "Re:Zero - Starting Life in Another World follows Subaru Natsuki, a young man who finds himself transported to a fantasy world where he discovers he has the ability to rewind time upon death. As he navigates this new world, he faces numerous challenges and uncovers dark secrets. The series masterfully blends fantasy, thriller, and psychological elements, exploring themes of despair, hope, and the struggle to change fate. Re:Zero has garnered critical acclaim for its complex narrative and character development.", "genre": "Fantasy, Thriller, Drama"},
    
    {"id": "anime29", "text": "Akame ga Kill! centers on Tatsumi, a young warrior who joins a group of assassins known as Night Raid to fight against a corrupt empire. The series features intense battles, memorable characters, and explores themes of justice, sacrifice, and the moral complexities of revolution. With its action-packed sequences and emotional storytelling, Akame ga Kill! captivates viewers with its blend of fantasy and social commentary, making it a popular choice among fans of shonen anime.", "genre": "Action, Adventure, Fantasy"},
    
    {"id": "anime30", "text": "Gintama is a unique blend of comedy, action, and science fiction set in an alternate history where aliens have invaded Japan. The story follows Gintoki Sakata, a lazy samurai who takes on odd jobs to make ends meet while navigating the absurdities of his world. Known for its humor, parodies of other anime, and character development, Gintama offers a refreshing take on the shonen genre. Its mix of serious arcs and comedic episodes makes it a beloved classic among anime fans.", "genre": "Action, Comedy, Sci-Fi"}
]