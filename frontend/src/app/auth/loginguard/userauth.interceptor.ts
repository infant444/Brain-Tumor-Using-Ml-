import { HttpEvent, HttpHandler, HttpInterceptor, HttpRequest } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { UserService } from '../../Services/user/user.service';


@Injectable()
export class userauthInterceptor implements HttpInterceptor{
  constructor(private userservice:UserService){}
  intercept(req: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    const user=this.userservice.currentUser;
    console.log(user)
    if(user.token){
      req=req.clone({
        setHeaders:{
          access_token:user.token
        }
      })
    }
    return next.handle(req)
  }
}
