CREATE TABLE result (
    id INT(255) AUTO_INCREMENT PRIMARY KEY,
    name VARCHAR(255) COLLATE utf8mb4_general_ci NOT NULL,
    height_pixel INT(255) NOT NULL,
    width_pixel INT(255) NOT NULL,
    height_cm INT(255) NOT NULL,
    width_cm INT(255) NOT NULL,
    classification VARCHAR(255) COLLATE utf8mb4_general_ci NOT NULL,
    created_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6),
    updated_at TIMESTAMP(6) DEFAULT CURRENT_TIMESTAMP(6) ON UPDATE CURRENT_TIMESTAMP(6)
);
