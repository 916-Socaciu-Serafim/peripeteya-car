import socket

server_socket = socket.socket()
server_socket.bind(('0.0.0.0', 8000))

server_socket.listen(0)

client_socket = server_socket.accept()[0]

while True:
    data = client_socket.recv(1024).decode()
    if data:
        print("Message received:", data)
        message_to_send = input("send back>>>")
        server_socket.send(message_to_send.encode())
