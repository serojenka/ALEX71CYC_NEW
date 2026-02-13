
        if (full_match) {
            // Found it! Store result atomically
            if (atomicCAS((int*)&d_found_flag, 0, 1) == 0) {
                d_result.found = true;
                d_result.private_key = key;
                for (int j = 0; j < 20; j++) {
                    d_result.hash160[j] = h160[j];
                }
                for (int j = 0; j < 33; j++) {
                    d_result.pubkey[j] = pubkey[j];
                }
            }
            return;
        }
