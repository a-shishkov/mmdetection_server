SET GLOBAL sql_mode='NO_AUTO_VALUE_ON_ZERO'

CREATE DATABASE IF NOT EXISTS annotations;

use annotations;

CREATE TABLE IF NOT EXISTS images (
    id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    file_name VARCHAR(100) NOT NULL
);

CREATE TABLE annotations (
    id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    --   category_id INT NOT NULL FOREIGN KEY REFERENCES categories(id),
    category_id INT NOT NULL,
    image_id INT NOT NULL,
    FOREIGN KEY (image_id) REFERENCES images(id)
);

CREATE TABLE IF NOT EXISTS segments (
    id INT NOT NULL PRIMARY KEY AUTO_INCREMENT,
    segment INT NOT NULL,
    annotation_id INT NOT NULL,
    FOREIGN KEY (annotation_id) REFERENCES annotations(id)
);