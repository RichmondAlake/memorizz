-- A persona's role is free text ("Night-shift harbour pilot who trains the
-- junior pilots"), and personas.role_type was created 50 bytes wide: a longer
-- role failed the whole agent save with ORA-12899. The provider widens the
-- column at startup; this is the same change for a schema managed by hand.
-- Safe to run more than once.

DECLARE
    current_length NUMBER;
    current_used VARCHAR2(1);
BEGIN
    SELECT char_length, char_used INTO current_length, current_used
    FROM user_tab_columns
    WHERE table_name = 'PERSONAS' AND column_name = 'ROLE_TYPE';

    IF current_length < 600 OR current_used <> 'C' THEN
        EXECUTE IMMEDIATE
            'ALTER TABLE personas MODIFY (role_type VARCHAR2(600 CHAR))';
    END IF;
EXCEPTION
    WHEN NO_DATA_FOUND THEN NULL;
END;
/

COMMIT;
