# Use an official Nginx runtime as a parent image. Nginx is a lightweight and high-performance web server.
FROM nginx:alpine

# Set the working directory inside the container to where Nginx serves files from.
WORKDIR /usr/share/nginx/html

# Remove the default Nginx welcome page.
RUN rm -f index.html

# Copy your application's HTML file from your project directory into the container's web root.
COPY sms.html .

# Inform Docker that the container will listen on port 80 at runtime.
EXPOSE 80