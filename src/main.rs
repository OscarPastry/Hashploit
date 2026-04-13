//CLI input → Load wordlist → Flatten data → Send to GPU →
//Run OpenCL kernel → Compare hashes → Return match
use ocl :: {Buffer, Context, Device, Kernel, Platform, Program, Queue, flags}; //ocl - The OpenCL                                                             
use clap :: Parser; //handles CLI arguments 
use std :: fs; // read files from disk 

//CLI arguments
#[derive(Parser)]
#[command(name = "unhasher", about = "GPU-accelerated MD5 cracker")]
struct Args{
    #[arg(short = 'H', long)]
    hash: String,    // the MD5 hash you want to crack
    //./program -H abc123
    //OR
    //./program --Hash abc123
    #[arg(short, long)]
    wordlist: String, // path to your wordlist file 
    
}

struct Wordlist {
    words: Vec<String>, //original words ["cat", "dog", "hello"]
    flat: Vec<u8>, //all words as one big byte array [c,a,t,d,o,g,h,e,l,l,o]
    offsets: Vec<i32>, //where each word starts
    lengths: Vec<i32>, // how long is each word
}

fn load_wordlist(path: &str) -> Wordlist {
    let content = fs:: read_to_string(path)
        .expect("Failed to read wordList");

    let words: Vec<String> = content
        .lines()
        .filter(|l| !l.is_empty() && l.len() <= 55) // Removes empty lines , limits word lengths to
                                                    // 55 , trims whitespaces
        .map(|l| l.trim().to_string())
        .collect();

    let mut flat : Vec<u8> = Vec::new();
    let mut offsets : Vec<i32> = Vec::new();
    let mut lengths : Vec<i32> = Vec:: new();
    let mut offset = 0i32;// offset is init into a 32 bit int with value 0

    for word in &words {
        let bytes = word.as_bytes(); // converts &str -> & u8
        offsets.push(offset); // stores the current postion of word in flat vector
        lengths.push(bytes.len() as i32); // stores the length as a i32
        flat.extend_from_slice(bytes); // Appends all bytes of the current word to the flat
        offset += bytes.len() as i32; // Updates the offset for the next word
    }

    Wordlist {words, flat, offsets, lengths}
}

fn crack(target_hex: &str, wordlist_path: &str) -> ocl :: Result <()>{//returns Ok(()) or Error
    // Parse target hash
    let target_bytes = hex::decode(target_hex)
        .expect("Invalid hex hash");
    assert_eq!(target_bytes.len(), 16, "Must be an MD5 hash(32 hex chars)");
    
    let wl = load_wordlist(wordlist_path);
    let num_words = wl.words.len();
    println!("[*] Loaded {} words", num_words);
    println!("[*] Target: {}", target_hex);
    
    //Pick Intel Arc GPU
    //Lists all OpenCL platforms and picks the Intel one
    let platform = Platform::list()
        .into_iter() // loop 
        .find(|p| { 
            p.name()
                .unwrap_or_default()
                .to_lowercase()
                .contains("intel")
        })
        .unwrap_or_else(|| Platform::default()); //this is safeguard 
    
    let device = Device::first(platform)?; // ? operator :- If you find a device, give it to me; if you don't, 
                                           // return an error immediately and exit the function. 
    println!("[*] Using device: {}", device.name()?);

    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?; // You can’t share memory buffers or kernels between different contexts. 
                  // Everything inside this context belongs to that specific hardware setup you identified in the previous steps.
    let queue = Queue::new(&context, device, None)?; // [None] refer to Queue Properties 
    
    //Load Kernel source
    let kernel_src = fs::read_to_string("kernel/crack.cl")
        .expect("Failed to read crack.cl");
    let program = Program::builder()
        .src(kernel_src)
        .build(&context)?;

    //allocate GPU buffers
    let flat_buf = Buffer::<u8>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(wl.flat.len())
        .copy_host_slice(&wl.flat)  //copying from RAM to VRAM
        .build()?;

    let offsets_buf = Buffer::<i32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(wl.offsets.len())
        .copy_host_slice(&wl.offsets)
        .build()?;

    let lengths_buf = Buffer::<i32>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(wl.lengths.len())
        .copy_host_slice(&wl.lengths)
        .build()?;
    let target_buf = Buffer::<u8>::builder()
        .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(target_bytes.len())
        .copy_host_slice(&target_bytes)
        .build()?;
    
    let result_buf = Buffer::<i32>::builder()
    .queue(queue.clone())
        .flags(flags::MEM_READ_ONLY | flags::MEM_COPY_HOST_PTR)
        .len(1)
        .copy_host_slice(&[-1i32])
        .build()?;
    //build and launch the kernel
    
    let kernel = Kernel::builder()
        .program(&program)
        .name("crack")
        .queue(queue.clone())
        .global_work_size(num_words)
        .arg(&flat_buf)
        .arg(&offsets_buf)
        .arg(&lengths_buf)
        .arg(&target_buf)
        .arg(&result_buf)
        .build()?;

    println!("[*] launching {} GPU threads...", num_words);
    let start = std::time::Instant::now();

    unsafe { kernel.enq()?; } 
    queue.finish()?;

    let elaspsed = start.elapsed();

    let mut result = vec![-1i32];
    result_buf.read(&mut result).enq()?;

    let speed = num_words as f32 / elaspsed.as_secs_f32()/1_000_000.0;
    println!("[*] speed: {:.2}MH/s | Time: {:.2?}",speed,elaspsed);

    if result[0] != -1{
        println!("\n[*] CRACKED: {}", wl.words[result[0] as usize]);
    }else {
        println!("\n[-] NOT FOUND in wordlist");
    }

    Ok(())
}

fn main(){
    let args = Args::parse();
    crack(&args.hash, &args.wordlist).expect("Crack failed");
}

