import { TestBed } from '@angular/core/testing';

import { BraintumorService } from './braintumor.service';

describe('BraintumorService', () => {
  let service: BraintumorService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(BraintumorService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
