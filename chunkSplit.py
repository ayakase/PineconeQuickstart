# def chunk_text_file(filename, chunk_size=400):
#     # Read the file contents
#     with open(filename, 'r') as file:
#         text = file.read()

#     # Split the text into words
#     words = text.split()

#     # Break into chunks of 200 words each
#     chunks = []
#     for i in range(0, len(words), chunk_size):
#         chunk = ' '.join(words[i:i + chunk_size])
#         chunks.append({
#             "vol": 1,
#             "chapter":1,
#             "chunk": chunk
#             })

#     # Print the number of chunks
#     print(f"Total number of chunks: {len(chunks)}")
#     print(chunks[10])
#     return chunks

# # Use the function
# filename = 'vol1chap1.txt'  # Replace with your file name
# chunks = chunk_text_file(filename)
