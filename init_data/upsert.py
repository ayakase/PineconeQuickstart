from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import os
load_dotenv()
pinecone_key = os.getenv('SECRET_KEY')
pc = Pinecone(api_key=pinecone_key)
index_name = "itfield"
data = [
    {"id": "vec1", "text": "Frontend development focuses on the visual aspects of a website or application. It involves using HTML, CSS, and JavaScript to create interactive interfaces. Modern frameworks like React, Vue, and Angular are popular for building dynamic UIs, allowing developers to create seamless user experiences."},
    {"id": "vec2", "text": "Backend development refers to server-side programming that manages database interactions, user authentication, and server logic. Common languages for backend development include Python, Java, Ruby, and PHP. Developers use frameworks like Express, Django, and Spring to streamline the development process."},
    {"id": "vec3", "text": "Fullstack developers possess skills in both frontend and backend technologies. They can work on the entire application stack, from designing user interfaces to managing server and database functionality. This versatility allows them to understand the entire workflow and create cohesive applications."},
    {"id": "vec4", "text": "Responsive web design is crucial in today's mobile-first world. It ensures websites function well on various devices and screen sizes. Techniques include fluid grids, flexible images, and media queries, allowing developers to create visually appealing and user-friendly designs across platforms."},
    {"id": "vec5", "text": "Version control systems like Git are essential for managing code changes and collaborating in development teams. They allow developers to track revisions, revert to previous states, and work concurrently without conflicts. Platforms like GitHub and GitLab enhance collaboration and project management."},
    {"id": "vec6", "text": "APIs (Application Programming Interfaces) are critical in modern software development, enabling different applications to communicate. RESTful APIs and GraphQL are common choices for building web services. Developers use them to fetch or send data between the frontend and backend efficiently."},
    {"id": "vec7", "text": "JavaScript frameworks such as React and Vue enable developers to build dynamic, single-page applications (SPAs) with rich user interfaces. These frameworks enhance performance and maintainability through component-based architecture, allowing developers to create reusable components and manage application state effectively."},
    {"id": "vec8", "text": "Database management is vital for backend developers, with options like SQL and NoSQL databases. SQL databases, like MySQL and PostgreSQL, provide structured data storage, while NoSQL databases, like MongoDB and Cassandra, offer flexibility for handling unstructured data and scalability."},
    {"id": "vec9", "text": "DevOps integrates development and operations to improve software delivery speed and quality. It emphasizes automation, continuous integration, and continuous deployment (CI/CD). Tools like Docker, Jenkins, and Kubernetes streamline processes, enabling teams to deploy applications reliably and efficiently."},
    {"id": "vec10", "text": "Testing is a critical aspect of software development. Unit testing, integration testing, and end-to-end testing ensure code reliability and performance. Frameworks like Jest for JavaScript and PyTest for Python help developers identify bugs early in the development cycle."},
    {"id": "vec11", "text": "User Experience (UX) design is essential for creating applications that meet user needs. It involves research, prototyping, and usability testing to ensure that products are intuitive and enjoyable to use. Collaboration between UX designers and developers leads to successful applications."},
    {"id": "vec12", "text": "Security is a top priority in software development. Best practices include data encryption, input validation, and regular security audits. Developers must stay informed about vulnerabilities and incorporate security measures throughout the development lifecycle to protect user data and maintain trust."},
    {"id": "vec13", "text": "Cloud computing has transformed the way applications are developed and deployed. Services like AWS, Azure, and Google Cloud offer scalable infrastructure, enabling developers to host applications with ease. Cloud technologies facilitate collaboration and improve accessibility for distributed teams."},
    {"id": "vec14", "text": "Microservices architecture breaks applications into smaller, manageable services that communicate over APIs. This approach enhances scalability and resilience, allowing teams to develop and deploy services independently. It encourages agility and helps organizations adapt quickly to changing requirements."},
    {"id": "vec15", "text": "Mobile app development requires specific considerations for performance and usability. Developers must understand platform guidelines for iOS and Android, optimizing for battery life and network usage. Frameworks like React Native and Flutter enable cross-platform development, enhancing reach and efficiency."},
    {"id": "vec16", "text": "Accessibility in web design ensures that applications are usable by everyone, including individuals with disabilities. Developers should adhere to WCAG guidelines, implement ARIA roles, and test applications with assistive technologies. Creating accessible applications not only broadens the user base but also fosters inclusivity."},
    {"id": "vec17", "text": "State management is crucial in frontend development, particularly in large applications. Libraries like Redux and Vuex help manage application state predictably, allowing for easier debugging and testing. They ensure that UI components stay in sync with the underlying data model."},
    {"id": "vec18", "text": "Serverless architecture allows developers to build applications without managing server infrastructure. By using services like AWS Lambda and Azure Functions, developers can focus on writing code while the cloud provider automatically handles scaling, maintenance, and availability."},
    {"id": "vec19", "text": "Continuous learning is essential in the IT field, given its rapid evolution. Developers should engage in online courses, attend conferences, and participate in coding challenges to stay updated with the latest technologies and best practices, ensuring their skills remain relevant."},
    {"id": "vec20", "text": "Data analytics and visualization tools help developers and businesses make data-driven decisions. Technologies like Tableau, Power BI, and D3.js enable users to analyze data trends and present insights in visually appealing formats, enhancing understanding and engagement."},
    {"id": "vec21", "text": "Graphical user interfaces (GUIs) enhance user interaction with applications. Developers use frameworks like Electron to build desktop applications with web technologies, combining the power of the web with native desktop functionalities for a seamless user experience."},
    {"id": "vec22", "text": "Progressive Web Apps (PWAs) combine the best features of web and mobile applications. They provide offline access, push notifications, and can be installed on devices. PWAs enhance user engagement and improve performance, offering a native app-like experience without the need for installation."},
    {"id": "vec23", "text": "Code review is an important practice in software development, promoting code quality and knowledge sharing. It allows team members to provide feedback, catch bugs, and discuss implementation strategies. Regular code reviews foster a culture of collaboration and continuous improvement."},
    {"id": "vec24", "text": "Technical documentation is essential for maintaining codebases and ensuring knowledge transfer among team members. Clear documentation helps onboard new developers and provides guidelines for using libraries and APIs, enhancing productivity and reducing misunderstandings."},
    {"id": "vec25", "text": "Containerization with Docker allows developers to package applications with their dependencies, ensuring consistent environments across development, testing, and production. This eliminates the 'it works on my machine' problem and simplifies deployment processes."},
    {"id": "vec26", "text": "API documentation is vital for developers who consume or provide APIs. Well-structured documentation includes usage examples, authentication methods, and response formats, enabling seamless integration and reducing errors during development."},
    {"id": "vec27", "text": "The Agile methodology emphasizes iterative development and collaboration. By breaking projects into manageable sprints, teams can adapt quickly to changes and deliver incremental value. Agile practices foster transparency, accountability, and customer feedback throughout the development process."},
    {"id": "vec28", "text": "Machine learning and artificial intelligence are increasingly integrated into software applications. Developers use frameworks like TensorFlow and PyTorch to build models that analyze data and make predictions. Understanding ML concepts is becoming essential for developers across various domains."},
    {"id": "vec29", "text": "Internet of Things (IoT) development involves creating applications that connect physical devices to the internet. This field requires knowledge of embedded systems, network protocols, and data management. IoT applications range from smart home devices to industrial automation solutions."},
    {"id": "vec30", "text": "Blockchain technology is transforming industries with its decentralized and secure transaction capabilities. Developers need to understand smart contracts, consensus algorithms, and cryptographic principles to create applications that leverage blockchain for various use cases, from finance to supply chain management."}
]

embeddings = pc.inference.embed(
    model="multilingual-e5-large",
    inputs=[d['text'] for d in data],
    parameters={"input_type": "passage", "truncate": "END"}
)
print(embeddings[0])

while not pc.describe_index(index_name).status['ready']:
    time.sleep(1)

index = pc.Index(index_name)

vectors = []
for d, e in zip(data, embeddings):
    vectors.append({
        "id": d['id'],
        "values": e['values'],
        "metadata": {'text': d['text']}
    })

index.upsert(
    vectors=vectors,
    namespace="ns1"
)