import { TestBed } from '@angular/core/testing';
import { CanActivateFn } from '@angular/router';

import { loginauthGuard } from './loginauth.guard';

describe('loginauthGuard', () => {
  const executeGuard: CanActivateFn = (...guardParameters) => 
      TestBed.runInInjectionContext(() => loginauthGuard(...guardParameters));

  beforeEach(() => {
    TestBed.configureTestingModule({});
  });

  it('should be created', () => {
    expect(executeGuard).toBeTruthy();
  });
});
