use rand::{RngCore, SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

pub struct Board {
    queens: u64,
}

pub fn get_counts() -> (usize, usize) {
    let mut solutions = Vec::new();
    gen_all_queens_recursive(0, 0, 0, &mut solutions);
    let res1 = solutions.len();

    let res2 = solutions
        .iter()
        .filter(|&s| validate_exclusion_map(*s, get_exclusion_map(*s)))
        .count();

    (res1, res2)
}

fn get_zone_map(queens: u64, exclusion_map: [u8; 64]) -> [u8; 64] {
    let mut zone_map = [0; 64];
    // TODO: Implement
    zone_map
}

// Gets the indices of the queens on the board (between 0 and 63)
fn get_queen_indices(queens: u64) -> [usize; 8] {
    let mut indices = [0; 8];
    for row in 0..8 {
        // Mask out all but the nth row
        let row_mask = 0b11111111 << (row * 8);
        let row_queens = queens & row_mask;
        // Find the column of the queen in this row
        let col = (row_queens >> (row * 8)).trailing_zeros() as usize;
        indices[row] = row * 8 + col;
    }
    indices
}

// A valid exclusion map is one where no square is zero, and every square
// except a square where a queen is place is excluded by at least two queens.
fn validate_exclusion_map(queens: u64, exclusion_map: [u8; 64]) -> bool {
    for i in 0..64 {
        let mask = exclusion_map[i];
        // No square is zero
        if mask == 0 {
            return false;
        }
        // If this is not a queen's square, it must be excluded by at least two queens
        if (queens & (1 << i)) == 0 {
            if mask.count_ones() < 2 {
                return false;
            }
        } else {
            // It must only be excluded by the queen itself
            if mask.count_ones() != 1 {
                return false;
            }
        }
    }
    true
}

// Returns a map of which squares on the grid are excluded by which queens.
// This is used to speed up the generation of boards.
fn get_exclusion_map(queens: u64) -> [u8; 64] {
    let mut exclusion_map = [0; 64];

    let mut qi = 0;
    for idx in 0..64 {
        if queens & (1 << idx) == 0 {
            // Skip non-queen squares
            continue;
        }

        exclusion_map[idx] |= 1 << qi;
        let row = idx / 8;
        let col = idx % 8;

        for i in 0..8 {
            exclusion_map[row * 8 + i] |= 1 << qi;
            exclusion_map[i * 8 + col] |= 1 << qi;
        }

        // Exclude the upper left and upper right diagonals
        if row > 0 {
            if col > 0 {
                exclusion_map[row * 8 + col - 9] |= 1 << qi;
            }
            if col < 7 {
                exclusion_map[row * 8 + col - 7] |= 1 << qi;
            }
        }

        // Exclude the lower left and lower right diagonals
        if row < 7 {
            if col > 0 {
                exclusion_map[row * 8 + col + 7] |= 1 << qi;
            }
            if col < 7 {
                exclusion_map[row * 8 + col + 9] |= 1 << qi;
            }
        }

        // Increment the queen index
        qi += 1;
    }

    exclusion_map
}

fn gen_random_queens() -> u64 {
    gen_random_queens_impl(None)
}

fn gen_random_queens_with_seed(seed: u64) -> u64 {
    gen_random_queens_impl(Some(seed))
}

// Checks if a queen can be placed at the specified position without violating the rules.
fn is_valid_queen_placement(current: u64, row: usize, col: usize, used_cols: u8) -> bool {
    (used_cols & (1 << col)) == 0
        && (row == 0
            || (col == 0 || (current & (1 << (row * 8 + col - 9)) == 0))
                && (col == 7 || (current & (1 << (row * 8 + col - 7)) == 0)))
}

// Places a queen at the specified position and returns the updated current and used_cols.
fn place_queen(current: u64, row: usize, col: usize, used_cols: u8) -> (u64, u8) {
    (current | (1 << (row * 8 + col)), used_cols | (1 << col))
}

fn gen_all_queens_recursive(current: u64, row: usize, used_cols: u8, results: &mut Vec<u64>) {
    if row == 8 {
        results.push(current);
        return;
    }

    for col in 0..8 {
        if is_valid_queen_placement(current, row, col, used_cols) {
            let (new_current, new_used_cols) = place_queen(current, row, col, used_cols);
            gen_all_queens_recursive(new_current, row + 1, new_used_cols, results);
        }
    }
}

fn gen_random_queens_impl(seed: Option<u64>) -> u64 {
    let mut rng = if let Some(seed) = seed {
        ChaCha8Rng::seed_from_u64(seed)
    } else {
        let mut inner = rand::rng();
        ChaCha8Rng::seed_from_u64(inner.next_u64())
    };

    // State for each row: which column was chosen, and the shuffled order of columns
    let mut cols_per_row: [Vec<usize>; 8] = Default::default();
    for row in 0..8 {
        let mut cols = (0..8).collect::<Vec<_>>();
        cols.shuffle(&mut rng);
        cols_per_row[row] = cols;
    }

    let mut row = 0;
    let mut current: u64 = 0;
    let mut used_cols: u8 = 0;
    let mut col_indices = [0usize; 8]; // Which index in the shuffled cols we're trying for each row

    while row < 8 {
        let mut placed = false;
        while col_indices[row] < 8 {
            let col = cols_per_row[row][col_indices[row]];
            if is_valid_queen_placement(current, row, col, used_cols) {
                let (new_current, new_used_cols) = place_queen(current, row, col, used_cols);
                current = new_current;
                used_cols = new_used_cols;
                placed = true;
                col_indices[row] += 1;
                break;
            }
            col_indices[row] += 1;
        }
        if placed {
            row += 1;
            if row < 8 {
                // Prepare for the next row
                if cols_per_row[row].is_empty() {
                    let mut cols = (0..8).collect::<Vec<_>>();
                    cols.shuffle(&mut rng);
                    cols_per_row[row] = cols;
                }
                col_indices[row] = 0;
            }
        } else {
            // Backtrack
            if row == 0 {
                // Should never happen, but just in case
                panic!("Could not place 8 queens with the given seed");
            }
            row -= 1;
            // Remove the queen from the previous row
            let prev_col = cols_per_row[row][col_indices[row] - 1];
            current &= !(1 << (row * 8 + prev_col));
            used_cols &= !(1 << prev_col);
        }
    }

    current
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_queens(board: u64) -> u32 {
        board.count_ones()
    }

    fn is_valid_queen_placement(board: u64) -> bool {
        // Check each queen's position against all other queens
        for row1 in 0..8 {
            for col1 in 0..8 {
                let idx1 = row1 * 8 + col1;
                if (board & (1 << idx1)) == 0 {
                    continue; // No queen at this position
                }

                // Check against all other queens
                for row2 in 0..8 {
                    for col2 in 0..8 {
                        let idx2 = row2 * 8 + col2;
                        if idx1 == idx2 || (board & (1 << idx2)) == 0 {
                            continue; // Same position or no queen
                        }

                        // Check same row
                        if row1 == row2 {
                            return false;
                        }
                        // Check same column
                        if col1 == col2 {
                            return false;
                        }
                        // Check diagonal adjacency (only adjacent diagonals are invalid)
                        let row_diff = (row1 as i32 - row2 as i32).abs();
                        let col_diff = (col1 as i32 - col2 as i32).abs();
                        if row_diff == 1 && col_diff == 1 {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    #[test]
    fn test_gen_all_queens_recursive() {
        let mut solutions = Vec::new();
        gen_all_queens_recursive(0, 0, 0, &mut solutions);

        assert_eq!(solutions.len(), 5242);

        // Test 1: Verify each solution has exactly 8 queens
        for solution in &solutions {
            assert_eq!(
                count_queens(*solution),
                8,
                "Each solution should have exactly 8 queens"
            );
        }

        // Test 2: Verify each solution follows the rules
        for solution in &solutions {
            assert!(
                is_valid_queen_placement(*solution),
                "Solution should follow queen placement rules"
            );
        }

        // Test 3: Verify all solutions are unique
        let mut unique_solutions = std::collections::HashSet::new();
        for solution in &solutions {
            assert!(
                unique_solutions.insert(*solution),
                "All solutions should be unique"
            );
        }

        // Print number of solutions found
        println!("Found {} valid solutions", solutions.len());
    }

    #[test]
    fn test_get_exclusion_map() {
        // Test with a single queen in the middle
        let queens = 1 << (3 * 8 + 3); // Queen at (3,3)
        let exclusion_map = get_exclusion_map(queens);

        // Check that the queen's position is marked
        assert_eq!(
            exclusion_map[3 * 8 + 3],
            1,
            "Queen's position should be marked"
        );

        // Check row exclusions
        for col in 0..8 {
            assert_eq!(exclusion_map[3 * 8 + col], 1, "Row should be marked");
        }

        // Check column exclusions
        for row in 0..8 {
            assert_eq!(exclusion_map[row * 8 + 3], 1, "Column should be marked");
        }

        // Check diagonal exclusions (only adjacent squares)
        assert_eq!(
            exclusion_map[2 * 8 + 2],
            1,
            "Upper left diagonal should be marked"
        );
        assert_eq!(
            exclusion_map[2 * 8 + 4],
            1,
            "Upper right diagonal should be marked"
        );
        assert_eq!(
            exclusion_map[4 * 8 + 2],
            1,
            "Lower left diagonal should be marked"
        );
        assert_eq!(
            exclusion_map[4 * 8 + 4],
            1,
            "Lower right diagonal should be marked"
        );

        // Check that non-excluded squares are not marked
        assert_eq!(
            exclusion_map[0 * 8 + 0],
            0,
            "Non-excluded square should not be marked"
        );
        assert_eq!(
            exclusion_map[7 * 8 + 7],
            0,
            "Non-excluded square should not be marked"
        );

        // Test with multiple queens
        let queens = (1 << (0 * 8 + 0)) | (1 << (7 * 8 + 7)); // Queens at (0,0) and (7,7)
        let exclusion_map = get_exclusion_map(queens);

        // Check that both queens' positions are marked
        assert_eq!(
            exclusion_map[0 * 8 + 0],
            1,
            "First queen's position should be marked"
        );
        assert_eq!(
            exclusion_map[7 * 8 + 7],
            2,
            "Second queen's position should be marked"
        );

        // Check that squares excluded by both queens have both bits set
        assert_eq!(
            exclusion_map[0 * 8 + 7],
            3,
            "Square excluded by both queens should have both bits set"
        );
        assert_eq!(
            exclusion_map[7 * 8 + 0],
            3,
            "Square excluded by both queens should have both bits set"
        );

        // Check that squares excluded by only one queen have only one bit set
        assert_eq!(
            exclusion_map[0 * 8 + 1],
            1,
            "Square excluded by only first queen should have only first bit set"
        );
        assert_eq!(
            exclusion_map[7 * 8 + 6],
            2,
            "Square excluded by only second queen should have only second bit set"
        );
    }

    #[test]
    fn test_validate_exclusion_map() {
        // Valid case: use a full 8-queen solution
        let mut solutions = Vec::new();
        gen_all_queens_recursive(0, 0, 0, &mut solutions);
        let queens = solutions[0];
        let exclusion_map = get_exclusion_map(queens);
        assert!(
            validate_exclusion_map(queens, exclusion_map),
            "Valid exclusion map for full 8-queen solution should return true"
        );

        // Invalid case: zero in exclusion map
        let mut exclusion_map = get_exclusion_map(queens);
        exclusion_map[10] = 0;
        assert!(
            !validate_exclusion_map(queens, exclusion_map),
            "Exclusion map with zero should return false"
        );

        // Invalid case: non-queen square not excluded by at least two queens
        let mut exclusion_map = get_exclusion_map(queens);
        // Check if there are any non-queen squares
        if let Some(non_queen_idx) = (0..64).find(|&i| (queens & (1 << i)) == 0) {
            exclusion_map[non_queen_idx] = 1; // Only first queen
            assert!(
                !validate_exclusion_map(queens, exclusion_map),
                "Non-queen square with only one exclusion should return false"
            );
        }

        // Valid case: queen square is allowed to have only one bit set
        let mut exclusion_map = [1u8; 64];
        let queens = 1 << 0;
        exclusion_map[0] = 1; // Queen at (0,0)
        for i in 1..64 {
            exclusion_map[i] = 3; // Excluded by two queens (even though only one is present)
        }
        assert!(
            validate_exclusion_map(queens, exclusion_map),
            "Queen square with one bit set is valid"
        );
    }

    #[test]
    fn test_gen_random_queens() {
        // Test multiple random generations to ensure they're different
        let solution1 = gen_random_queens();
        let solution2 = gen_random_queens();
        let solution3 = gen_random_queens();

        // Verify each solution has exactly 8 queens
        assert_eq!(
            count_queens(solution1),
            8,
            "First solution should have 8 queens"
        );
        assert_eq!(
            count_queens(solution2),
            8,
            "Second solution should have 8 queens"
        );
        assert_eq!(
            count_queens(solution3),
            8,
            "Third solution should have 8 queens"
        );

        // Verify each solution follows the rules
        assert!(
            is_valid_queen_placement(solution1),
            "First solution should be valid"
        );
        assert!(
            is_valid_queen_placement(solution2),
            "Second solution should be valid"
        );
        assert!(
            is_valid_queen_placement(solution3),
            "Third solution should be valid"
        );

        // Verify solutions are different (not guaranteed but very likely)
        assert_ne!(solution1, solution2, "Random solutions should be different");
        assert_ne!(solution2, solution3, "Random solutions should be different");
        assert_ne!(solution1, solution3, "Random solutions should be different");
    }

    #[test]
    fn test_gen_random_queens_with_seed() {
        // Test that same seed produces same result
        let seed = 12345;
        let solution1 = gen_random_queens_with_seed(seed);
        let solution2 = gen_random_queens_with_seed(seed);

        // Verify solutions are identical
        assert_eq!(
            solution1, solution2,
            "Same seed should produce same solution"
        );

        // Verify solution has exactly 8 queens
        assert_eq!(count_queens(solution1), 8, "Solution should have 8 queens");

        // Verify solution follows the rules
        assert!(
            is_valid_queen_placement(solution1),
            "Solution should be valid"
        );

        // Test that different seeds produce different results
        let solution3 = gen_random_queens_with_seed(seed + 1);
        assert_ne!(
            solution1, solution3,
            "Different seeds should produce different solutions"
        );
    }

    #[test]
    fn test_random_queens_are_valid_and_in_all_solutions() {
        use std::collections::HashSet;

        // Generate all valid boards
        let mut all_solutions = Vec::new();
        gen_all_queens_recursive(0, 0, 0, &mut all_solutions);
        let all_solutions_set: HashSet<u64> = all_solutions.into_iter().collect();

        let mut rng = rand::rng();
        for _ in 0..10000 {
            let seed: u64 = rng.next_u64();
            let board = gen_random_queens_with_seed(seed);
            // Check that the board is valid
            assert!(is_valid_queen_placement(board), "Random board is not valid");
            // Check that the exclusion map is valid
            let exclusion_map = get_exclusion_map(board);
            assert!(
                validate_exclusion_map(board, exclusion_map),
                "Exclusion map is not valid"
            );
            // Check that the board is in the set of all valid boards
            assert!(
                all_solutions_set.contains(&board),
                "Random board not in set of all valid boards"
            );
        }
    }

    #[test]
    fn test_get_queen_indices() {
        // Test with a known valid solution
        let mut solutions = Vec::new();
        gen_all_queens_recursive(0, 0, 0, &mut solutions);
        let test_board = solutions[0];

        let indices = get_queen_indices(test_board);

        // Verify we got exactly 8 indices
        assert_eq!(indices.len(), 8, "Should return exactly 8 indices");

        // Verify each index is within bounds (0-63)
        for &idx in &indices {
            assert!(idx < 64, "Index {} should be less than 64", idx);
        }

        // Verify each index corresponds to a queen in the original board
        for &idx in &indices {
            assert!(
                (test_board & (1 << idx)) != 0,
                "Index {} should correspond to a queen in the board",
                idx
            );
        }

        // Verify indices are in ascending order (since we process rows sequentially)
        for i in 1..8 {
            assert!(
                indices[i] > indices[i - 1],
                "Indices should be in ascending order"
            );
        }

        // Verify each row has exactly one queen
        let mut row_counts = [0; 8];
        for &idx in &indices {
            let row = idx / 8;
            row_counts[row] += 1;
        }
        for count in row_counts {
            assert_eq!(count, 1, "Each row should have exactly one queen");
        }

        // Verify that each column has exactly one queen
        let mut col_counts = [0; 8];
        for &idx in &indices {
            let col = idx % 8;
            col_counts[col] += 1;
        }
        for count in col_counts {
            assert_eq!(count, 1, "Each column should have exactly one queen");
        }

        // Test with a manually constructed board
        let manual_board =
            0b00000001_00000010_00000100_00001000_00010000_00100000_01000000_10000000;
        let manual_indices = get_queen_indices(manual_board);
        let expected_indices = [7, 14, 21, 28, 35, 42, 49, 56];
        assert_eq!(
            manual_indices, expected_indices,
            "Manual board indices don't match expected"
        );
    }
}
