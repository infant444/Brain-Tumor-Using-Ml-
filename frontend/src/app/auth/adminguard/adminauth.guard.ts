import { Injectable } from '@angular/core';
import { ActivatedRouteSnapshot, CanActivate, Router, RouterStateSnapshot, UrlTree } from '@angular/router';
import { Observable } from 'rxjs';
import { UserService } from '../../Services/user/user.service';

@Injectable()
export class AdminauthGuard implements CanActivate{

  constructor(private userservices:UserService,private router:Router){}

  canActivate(route: ActivatedRouteSnapshot, state: RouterStateSnapshot): boolean | UrlTree | Observable<boolean | UrlTree> | Promise<boolean | UrlTree> {
   if(this.userservices.currentUser.admin){
    return true
   }
   this.router.navigate(['/login'],{queryParams:{returnUrl:state.url}})
   return false
  }


}
