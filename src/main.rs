mod board;

fn main() {
    let (res1, res2) = board::get_counts();
    println!("{}", res1);
    println!("{}", res2);
}
