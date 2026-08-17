-- ================================================
-- Student Performance SQL Analysis
-- ================================================


-- 1. View first 10 students
SELECT *
FROM students
LIMIT 10;


-- 2. Total number of students
SELECT COUNT(*) AS total_students
FROM students;


-- 3. Pass and fail distribution
SELECT
    performance,
    COUNT(*) AS student_count
FROM students
GROUP BY performance;


-- 4. Average final score
SELECT
    ROUND(AVG(final_score), 2) AS average_final_score
FROM students;


-- 5. Average final score by gender
SELECT
    gender,
    COUNT(*) AS student_count,
    ROUND(AVG(final_score), 2) AS average_score
FROM students
GROUP BY gender;


-- 6. Average score by parental education
SELECT
    parental_education,
    COUNT(*) AS student_count,
    ROUND(AVG(final_score), 2) AS average_score
FROM students
GROUP BY parental_education
ORDER BY average_score DESC;


-- 7. Average attendance by performance
SELECT
    performance,
    ROUND(AVG(attendance_percentage), 2)
        AS average_attendance
FROM students
GROUP BY performance;


-- 8. Average study hours by performance
SELECT
    performance,
    ROUND(AVG(study_hours), 2)
        AS average_study_hours
FROM students
GROUP BY performance;


-- 9. Average absences by performance
SELECT
    performance,
    ROUND(AVG(absences), 2)
        AS average_absences
FROM students
GROUP BY performance;


-- 10. Top 10 students by final score
SELECT
    student_id,
    name,
    final_score,
    performance
FROM students
ORDER BY final_score DESC
LIMIT 10;


-- 11. Students with attendance below 60%
SELECT
    student_id,
    name,
    attendance_percentage,
    final_score,
    performance
FROM students
WHERE attendance_percentage < 60
ORDER BY attendance_percentage;


-- 12. Students at high risk
SELECT
    student_id,
    name,
    study_hours,
    attendance_percentage,
    previous_score,
    absences,
    performance
FROM students
WHERE attendance_percentage < 70
   OR study_hours < 2
   OR absences > 20
ORDER BY attendance_percentage;